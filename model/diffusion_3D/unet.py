import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=4, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, depth, height, width = input.shape
        n_head = self.n_head

        norm = self.norm(input)
        # print(norm.shape)
        qkv = self.qkv(norm)
        # print(qkv.shape)
        q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        return out + input


class SelfAttention_fuse(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(4, in_channel)
        self.out = nn.Conv3d(in_channel, in_channel, 1)
        self.defmgen=nn.Conv3d(in_channel,3,3,padding=1)
        self.nonlinear=nn.Conv3d(3, 3, 3, padding=1)

    def forward(self, q,k,v,size):
        batch, channel, depth, height, width = q.shape

        n_head = self.n_head
        residual=q
        norm_q = self.norm(q)
        norm_k = self.norm(k)
        norm_v = self.norm(v)
        

        qkv=torch.cat([norm_q,norm_k,norm_v],dim=1)
        q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        out=self.defmgen(out+residual)
        out=F.upsample_nearest(out,size)
        out=self.nonlinear(out)
        return out


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128,
        opt=None
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
        self.opt=opt
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size[1]
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               time_emb_dim=time_dim, dropout=dropout, with_attn=False)
        ])


        ups_diff = []
        ups_regis = []
        ups_adapt=[]
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                feat_channel=feat_channels.pop()
                ups_diff.append(ResnetBlocWithAttn(
                    pre_channel+feat_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                regischannel=pre_channel+feat_channel+channel_mult
                ups_regis.append(ResnetBlocWithAttn(
                    regischannel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                ups_adapt.append(
                    SelfAttention_fuse(channel_mult)
                )
                pre_channel = channel_mult
            if not is_last:
                ups_adapt.append(nn.Identity())
                ups_diff.append(Upsample(pre_channel))
                ups_regis.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups_diff = nn.ModuleList(ups_diff)
        self.ups_regis = nn.ModuleList(ups_regis)
        self.ups_adapt=nn.ModuleList(ups_adapt)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel))
        # self.final_attn=SelfAttention_fuse(1)
        self.final_conv_defm = Block(pre_channel+1,3,groups=3)

    def forward(self, x,x_m, time):
        input_size=(x.size(2),x.size(3),x.size(4))
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        x_1=x
        x_2=x
        defm=[]
        # x_1vis=[]
        for layerd,layerr,layera in zip(self.ups_diff,self.ups_regis,self.ups_adapt):
            if isinstance(layerd, ResnetBlocWithAttn):
                feat=feats.pop()
                x_1 = layerd(torch.cat((x_1, feat), dim=1), t)
                x_2 = layerr(torch.cat((x_2, feat,x_1), dim=1), t)
                defm_=layera(x_2,x_1,x_1,input_size)
                defm.append(defm_)
            else:
                x_1 = layerd(x_1)
                x_2 = layerr(x_2)
        recon=self.final_conv(x_1)
        defm=torch.stack(defm,dim=1)
        defm=torch.cat([defm,self.final_conv_defm(torch.cat((x_2,recon), dim=1)).unsqueeze_(1)],dim=1)
        defm=torch.mean(defm,dim=1)
        return recon,defm

class Dense3DSpatialTransformer(nn.Module):
    def __init__(self):
        super(Dense3DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        input1 = (input1[:, :1] + 1) / 2.0
        return self._transform(input1, input2[:, 0], input2[:, 1], input2[:, 2])

    def _transform(self, input1, dDepth, dHeight, dWidth):
        batchSize = dDepth.shape[0]
        dpt = dDepth.shape[1]
        hgt = dDepth.shape[2]
        wdt = dDepth.shape[3]

        D_mesh, H_mesh, W_mesh = self._meshgrid(dpt, hgt, wdt)
        D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        D_upmesh = dDepth + D_mesh
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, D_upmesh, H_upmesh, W_upmesh)

    def _meshgrid(self, dpt, hgt, wdt):
        d_t = torch.linspace(0.0, dpt-1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt).to('cuda:0')
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).to('cuda:0')
        h_t = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).to('cuda:0')
        w_t = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        return d_t, h_t, w_t

    def _interpolate(self, input, D_upmesh, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        depth  = input.shape[2]
        height = input.shape[3]
        width  = input.shape[4]

        img = torch.zeros(nbatch, nch, depth+2,  height+2, width+2).to('cuda:0')
        img[:, :, 1:-1, 1:-1, 1:-1] = input

        imgDpt = img.shape[2]
        imgHgt = img.shape[3]  # 256+2 = 258
        imgWdt = img.shape[4]  # 256+2 = 258

        # D_upmesh, H_upmesh, W_upmesh = [D, H, W] -> [BDHW,]
        D_upmesh = D_upmesh.view(-1).float()+1.0  # (BDHW,)
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BDHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BDHW,)

        # D_upmesh, H_upmesh, W_upmesh -> Clamping into [0, 257] -- index
        df = torch.floor(D_upmesh).int()
        dc = df + 1
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        df = torch.clamp(df, 0, imgDpt-1)  # (BDHW,)
        dc = torch.clamp(dc, 0, imgDpt-1)  # (BDHW,)
        hf = torch.clamp(hf, 0, imgHgt-1)  # (BDHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BDHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BDHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BDHW,)

        # Find batch indexes
        rep = torch.ones([depth*height*width, ]).unsqueeze_(1).transpose(1, 0).to('cuda:0')
        bDHW = torch.matmul((torch.arange(0, nbatch).float()*imgDpt*imgHgt*imgWdt).unsqueeze_(1).to('cuda:0'), rep).view(-1).int()

        # Box updated indexes
        HW = imgHgt*imgWdt
        W = imgWdt
        # x: W, y: H, z: D
        idx_000 = bDHW + df*HW + hf*W + wf
        idx_100 = bDHW + dc*HW + hf*W + wf
        idx_010 = bDHW + df*HW + hc*W + wf
        idx_110 = bDHW + dc*HW + hc*W + wf
        idx_001 = bDHW + df*HW + hf*W + wc
        idx_101 = bDHW + dc*HW + hf*W + wc
        idx_011 = bDHW + df*HW + hc*W + wc
        idx_111 = bDHW + dc*HW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_000 = torch.index_select(img_flat, 0, idx_000.long())
        val_100 = torch.index_select(img_flat, 0, idx_100.long())
        val_010 = torch.index_select(img_flat, 0, idx_010.long())
        val_110 = torch.index_select(img_flat, 0, idx_110.long())
        val_001 = torch.index_select(img_flat, 0, idx_001.long())
        val_101 = torch.index_select(img_flat, 0, idx_101.long())
        val_011 = torch.index_select(img_flat, 0, idx_011.long())
        val_111 = torch.index_select(img_flat, 0, idx_111.long())

        dDepth  = dc.float() - D_upmesh
        dHeight = hc.float() - H_upmesh
        dWidth  = wc.float() - W_upmesh

        wgt_000 = (dWidth*dHeight*dDepth).unsqueeze_(1)
        wgt_100 = (dWidth * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_010 = (dWidth * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_110 = (dWidth * (1-dHeight) * (1-dDepth)).unsqueeze_(1)
        wgt_001 = ((1-dWidth) * dHeight * dDepth).unsqueeze_(1)
        wgt_101 = ((1-dWidth) * dHeight * (1-dDepth)).unsqueeze_(1)
        wgt_011 = ((1-dWidth) * (1-dHeight) * dDepth).unsqueeze_(1)
        wgt_111 = ((1-dWidth) * (1-dHeight) * (1-dDepth)).unsqueeze_(1)

        output = (val_000*wgt_000 + val_100*wgt_100 + val_010*wgt_010 + val_110*wgt_110 +
                  val_001 * wgt_001 + val_101 * wgt_101 + val_011 * wgt_011 + val_111 * wgt_111)
        output = output.view(nbatch, depth, height, width, nch).permute(0, 4, 1, 2, 3)  #B, C, D, H, W
        return output