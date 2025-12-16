from abc import ABC, abstractmethod
from datetime import timezone, datetime
from typing import Iterable

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO

from PIL import Image
from PIL.ImageFile import ImageFile
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchmetrics.functional.image import peak_signal_noise_ratio as torch_psnr
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as torch_msssim
from torchvision.transforms.v2.functional import to_tensor, to_pil_image

from src.data.base_dataset import BaseDataset
from src.data.kodak_dataset import KodakDataset
from src.data.transforms import RGBCompression, RGB_MIXED_LIC_MEAN, RGB_MIXED_LIC_STD
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config


class Codec(ABC):
    def __init__(self):
        self.qualities: Iterable | None = None
        self.label: str | None = None
        self.bpp = []
        self.psnr = []
        self.msssim = []

    @abstractmethod
    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float]:
        pass

    def eval_codec(self, dataset: BaseDataset, out_dir: str = None):
        for quality in self.qualities:
            avg_bpp = 0
            avg_psnr = 0
            avg_msssim = 0
            for i, (ref_img, _) in enumerate(dataset):
                if i == 18:
                    comp_img, bpp = self.encode(ref_img, quality)
                    ref_data = np.array(ref_img, dtype=np.float64)
                    comp_data = np.array(comp_img, dtype=np.float64)
                    psnr = sk_psnr(ref_data, comp_data, data_range=255)
                    torch_comp_img = to_tensor(comp_img).unsqueeze(dim=0)
                    torch_ref_img = to_tensor(ref_img).unsqueeze(dim=0)
                    msssim = torch_msssim(torch_comp_img, torch_ref_img, data_range=1.0).to("cpu").numpy().item()
                    full_out_dir = out_dir / self.label / str(quality)
                    full_out_dir.mkdir(parents=True, exist_ok=True)
                    ref_img.save(out_dir / self.label / str(quality) / f'ref_{i + 1}.png')
                    comp_img.save(out_dir / self.label / str(quality) / f'comp_{i + 1}.png')
                    avg_bpp += bpp
                    avg_psnr += psnr
                    avg_msssim += msssim
            # avg_bpp /= len(dataset)
            # avg_psnr /= len(dataset)
            # avg_msssim /= len(dataset)
            self.bpp.append(avg_bpp)
            self.psnr.append(avg_psnr)
            self.msssim.append(avg_msssim)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluated quality {quality} on codec {self.label}, "
                  f"bpp: {avg_bpp}, psnr: {avg_psnr}, msssim: {avg_msssim}")

class SwinLicCodec(Codec):
    def __init__(self):
        super().__init__()
        self.qualities = [
                "SWIN-B-IC_0.17.0",
                "SWIN-B-IC_0.17.0.1",
                "SWIN-B-IC_0.17.0.2",
                "SWIN-B-IC_0.17.0.3",
                "SWIN-B-IC_0.17.0.4",
                "SWIN-B-IC_0.17.0.5",
                "SWIN-B-IC_0.17.0.6",
                "SWIN-B-IC_0.17.0.7"
            ]
        self.label = "SwinLIC"

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float, float]:
        raise NotImplementedError

    def eval_codec(self, dataset: BaseDataset, out_dir: str = None):
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        dataloader = torch.utils.data.DataLoader(swin_test_dataset, batch_size=1, shuffle=False)

        for quality in self.qualities:
            avg_bpp = torch.tensor(0, dtype=torch.float32).to(device)
            avg_psnr = torch.tensor(0, dtype=torch.float32).to(device)
            avg_msssim = torch.tensor(0, dtype=torch.float32).to(device)
            model = model_from_config(
                {
                    "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
                    "weights": quality,
                    "args": {
                        "pretrained_encoder": False,
                        "encoder_type": "gdn_swin_v2_b",
                        "encoder_embed_dim": 128,
                        "encoder_dims": [128, 256, 512],
                        "encoder_depths": [2, 6, 24],
                        "encoder_num_heads": [4, 8, 16],
                        "encoder_window_size": [8, 8],
                        "encoder_sd_factor": 0.05,
                        "encoder_mlp_ratio": 4,
                        "encoder_dropout": 0,
                        "encoder_attention_dropout": 0,
                        "decoder_depths": [24, 6, 2],
                        "decoder_dims": [512, 512, 256, 128],
                        "decoder_num_heads": [16, 8, 4],
                        "decoder_window_size": [[8, 8], [8, 8], [8, 8]],
                        "decoder_mlp_ratio": [4, 4, 4],
                        "decoder_sd_factor": 0.03,
                        "decoder_dropout": 0,
                        "decoder_attention_dropout": 0,
                        "bottleneck_dim": 384,
                        "no_compress": False,
                        "checkpointing": False,
                    }
                })
            model.update()
            model.eval()
            model.to(device)
            with torch.no_grad():
                for i, (x_test_in, x_test) in enumerate(dataloader):
                    if i == 18:
                        x_test_in, x_test = x_test_in.to(device).detach(), x_test.to(device).detach()
                        compress_output = model.compress(x_test_in)
                        b_repr, shape = compress_output['strings'], compress_output['shape']
                        x_hat_test = model.decompress(b_repr, shape)['x_hat']

                        psnr = torch_psnr(x_test, x_hat_test, data_range=(0.0, 1.0), dim=(2, 3))
                        avg_psnr += psnr

                        msssim = torch_msssim(x_test, x_hat_test, data_range=1.0, reduction="elementwise_mean").mean()
                        avg_msssim += msssim

                        batch_size = len(b_repr[0])
                        bits = sum(len(stream) * 8 for group in b_repr for stream in group)
                        _, _, H, W = x_hat_test.shape
                        bpp = bits / (batch_size * H * W)
                        avg_bpp += bpp
                        ref_img: PIL.Image.Image = to_pil_image(x_test.detach().cpu().squeeze())
                        comp_img: PIL.Image.Image = to_pil_image(x_hat_test.detach().cpu().squeeze())
                        full_out_dir = out_dir / self.label / str(quality)
                        full_out_dir.mkdir(parents=True, exist_ok=True)
                        ref_img.save(out_dir / self.label / str(quality) / f'ref_{i + 1}.png')
                        comp_img.save(out_dir / self.label / str(quality) / f'comp_{i + 1}.png')

            # avg_bpp /= len(dataloader)
            # avg_psnr /= len(dataloader)
            # avg_msssim /= len(dataloader)
            self.bpp.append(avg_bpp.to("cpu").numpy())
            self.psnr.append(avg_psnr.to("cpu").numpy())
            self.msssim.append(avg_msssim.to("cpu").numpy())
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluated quality {quality} on codec {self.label}, "
                  f"bpp: {avg_bpp}, psnr: {avg_psnr}, msssim: {avg_msssim}")


class PILCodec(Codec, ABC):
    def __init__(self):
        super().__init__()
        self.qualities = range(0, 100, 1)

    def encode(self, ref_img: PIL.Image.Image, quality: int, out_dir: str = None, **kwargs) -> tuple[ImageFile, float]:
        with BytesIO() as img_bytes:
            ref_img.save(fp=img_bytes, quality=quality, **kwargs)
            raw_bytes = img_bytes.getvalue()
            bpp = (len(raw_bytes) * 8) / (ref_img.size[0] * ref_img.size[1])
            comp_img = Image.open(BytesIO(raw_bytes))
        return comp_img, bpp

class JPEGCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.qualities = range(0, 70, 1)
        self.label = 'JPEG'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[ImageFile, float]:
        return super().encode(ref_img=ref_img, quality=quality, subsampling=2, format='JPEG', **kwargs)

class WebPCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.qualities = range(0, 100, 1)
        self.label = 'WebP'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[ImageFile, float]:
        return super().encode(ref_img=ref_img, quality=quality, format='WebP', alpha_quality=0, **kwargs)

class AVIFCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.qualities = range(0, 70, 1)
        self.label = 'AVIF'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[ImageFile, float]:
        return super().encode(ref_img=ref_img, quality=quality, speed=0, format='AVIF', **kwargs)


jpeg_codec = JPEGCodec()
webp_codec = WebPCodec()
avif_codec = AVIFCodec()
swin_lic_codec = SwinLicCodec()
codecs = [
    jpeg_codec,
    webp_codec,
    avif_codec,
    swin_lic_codec
]

pil_test_dataset = KodakDataset(None, None, None, None)
swin_test_dataset = KodakDataset(RGBCompression(noresize=True, normalize=True, mean=RGB_MIXED_LIC_MEAN, std=RGB_MIXED_LIC_STD)
                                 ,None, None, None)

# jpeg_codec.psnr = [21.285822870858965, 21.287710097644485, 22.575647994171433, 25.597923873717004, 27.058256654461196, 27.826636520160054, 28.45069637021641, 29.082032072996636, 29.591863394080626, 29.972093531734263, 30.351398081737756, 30.67194812722823, 30.976363442542286, 31.20820448771574, 31.46466671888225, 31.70473517565847, 31.884465500500507, 32.07410348730027, 32.29723206812992, 32.45765553991972, 32.578221824481886, 32.729084535936664, 32.90301183744278, 33.017926200940714, 33.09730352060808, 33.25728334789894, 33.394340425262556, 33.4898501926375, 33.614900081159675, 33.772529784350816, 33.900947608922806, 34.041365773322056, 34.19695509925792, 34.3663496061228, 34.557412757658014, 34.74384199680334, 34.94802971175193, 35.19283808845624, 35.426208984426104, 35.7124553339385]
# jpeg_codec.msssim = [0.6453242301940918, 0.6453399658203125, 0.6905900239944458, 0.7932980060577393, 0.8459087610244751, 0.8694853782653809, 0.8875068426132202, 0.9031479954719543, 0.9156423211097717, 0.9240992665290833, 0.9318258762359619, 0.9373132586479187, 0.9430745840072632, 0.9461690783500671, 0.9502993226051331, 0.9534059166908264, 0.9559966325759888, 0.9578125476837158, 0.9613619446754456, 0.9625242948532104, 0.964087188243866, 0.9656962752342224, 0.9675755500793457, 0.9683647751808167, 0.9690638184547424, 0.9709205627441406, 0.9721788167953491, 0.9728404879570007, 0.9736270904541016, 0.9752240777015686, 0.9759318232536316, 0.9764483571052551, 0.9774438738822937, 0.978386402130127, 0.9795064926147461, 0.9805665016174316, 0.9815670251846313, 0.9827195405960083, 0.9836363792419434, 0.9848161339759827]
# jpeg_codec.bpp = [0.15751139322916666, 0.15755208333333334, 0.17364501953125, 0.20273844401041666, 0.23219807942708334, 0.2629191080729167, 0.2927652994791667, 0.3230183919270833, 0.3513997395833333, 0.3816324869791667, 0.4078572591145833, 0.4372151692708333, 0.46405029296875, 0.48846435546875, 0.5151570638020834, 0.5365193684895834, 0.56103515625, 0.5828653971354166, 0.6106974283854166, 0.63232421875, 0.6465250651041666, 0.67193603515625, 0.6946614583333334, 0.7163899739583334, 0.7324422200520834, 0.75262451171875, 0.7750447591145834, 0.7943522135416666, 0.8172200520833334, 0.8495279947916666, 0.8688761393229166, 0.8990071614583334, 0.9298095703125, 0.9654337565104166, 1.00830078125, 1.05047607421875, 1.0918986002604167, 1.1576944986979167, 1.19598388671875, 1.2789103190104167]
#
# swin_lic_codec.psnr = [36.04278, 35.5831, 34.932983, 34.168472, 32.703686, 31.403791, 29.892118, 28.729034]
# swin_lic_codec.msssim = [0.98831034, 0.98554945, 0.98144746, 0.9741574, 0.96025604, 0.94221437, 0.9186474, 0.8920996]
# swin_lic_codec.bpp = [0.9235026, 0.77905273, 0.6311849, 0.47851562, 0.34065756, 0.22273763, 0.146403, 0.10139974]
#
# avif_codec.psnr = [25.49522793843451, 26.223899609112724, 26.223899609112724, 26.740040598161933, 27.101302028118898, 27.101302028118898, 27.435748575251377, 27.435748575251377, 27.742629767650925, 27.95932065915632, 27.95932065915632, 28.327646547627523, 28.569943587303143, 28.569943587303143, 28.787288030200205, 28.787288030200205, 29.023478844347594, 29.241659125520066, 29.241659125520066, 29.518206934198112, 29.808451175855115, 29.808451175855115, 30.006261788136797, 30.006261788136797, 30.288582521560706, 30.516249833565503, 30.516249833565503, 30.807815684030754, 31.052719016292997, 31.052719016292997, 31.301494187993807, 31.616453534305148, 31.616453534305148, 31.88348751038166, 31.88348751038166, 32.160444865790645, 32.50637869299939, 32.50637869299939, 32.79144328887004, 33.106746523450894, 33.106746523450894, 33.46651059723746, 33.46651059723746, 33.75234117589646, 34.06135570626142, 34.06135570626142, 34.40236874636806, 34.70116677230311, 34.70116677230311, 35.002845761626375, 35.002845761626375, 35.341670289680074, 35.66274335076657, 35.66274335076657, 35.93306842555863, 36.199554981964056, 36.199554981964056, 36.507895474919295, 36.81223219761875, 36.81223219761875, 37.08624149130578, 37.08624149130578, 37.45440140559328, 37.66615802521938, 37.66615802521938, 37.89654550066071, 38.09561498942839, 38.09561498942839, 38.283902958478606, 38.283902958478606]
# avif_codec.msssim = [0.7986315488815308, 0.8197515606880188, 0.8197515606880188, 0.8326675295829773, 0.8435817956924438, 0.8435817956924438, 0.8545268774032593, 0.8545268774032593, 0.8641623258590698, 0.8693072199821472, 0.8693072199821472, 0.8793421983718872, 0.8872606754302979, 0.8872606754302979, 0.8922343254089355, 0.8922343254089355, 0.8984428644180298, 0.9038946628570557, 0.9038946628570557, 0.9098870158195496, 0.9160913825035095, 0.9160913825035095, 0.9208346605300903, 0.9208346605300903, 0.9258537292480469, 0.9298923015594482, 0.9298923015594482, 0.9345865249633789, 0.9383715987205505, 0.9383715987205505, 0.9415011405944824, 0.9462183117866516, 0.9462183117866516, 0.9495837092399597, 0.9495837092399597, 0.9530985951423645, 0.9567086100578308, 0.9567086100578308, 0.9599545001983643, 0.9627054929733276, 0.9627054929733276, 0.9660043120384216, 0.9660043120384216, 0.9678943753242493, 0.9703443050384521, 0.9703443050384521, 0.972688615322113, 0.9745067358016968, 0.9745067358016968, 0.9761876463890076, 0.9761876463890076, 0.9780670404434204, 0.9795264005661011, 0.9795264005661011, 0.9808334112167358, 0.9818735122680664, 0.9818735122680664, 0.9828916788101196, 0.98399418592453, 0.98399418592453, 0.9850016832351685, 0.9850016832351685, 0.9862717986106873, 0.9868833422660828, 0.9868833422660828, 0.9875279068946838, 0.9880558252334595, 0.9880558252334595, 0.9885677695274353, 0.9885677695274353]
# avif_codec.bpp = [0.024169921875, 0.029541015625, 0.029541015625, 0.033772786458333336, 0.03851318359375, 0.03851318359375, 0.043111165364583336, 0.043111165364583336, 0.047465006510416664, 0.051127115885416664, 0.051127115885416664, 0.0574951171875, 0.06412760416666667, 0.06412760416666667, 0.06899007161458333, 0.06899007161458333, 0.07489013671875, 0.08241780598958333, 0.08241780598958333, 0.08966064453125, 0.09828694661458333, 0.09828694661458333, 0.107177734375, 0.107177734375, 0.11794026692708333, 0.12801106770833334, 0.12801106770833334, 0.13972981770833334, 0.15081787109375, 0.15081787109375, 0.16365559895833334, 0.18363444010416666, 0.18363444010416666, 0.19805908203125, 0.19805908203125, 0.2164306640625, 0.23972574869791666, 0.23972574869791666, 0.2613321940104167, 0.2869873046875, 0.2869873046875, 0.3166707356770833, 0.3166707356770833, 0.33984375, 0.37286376953125, 0.37286376953125, 0.4065348307291667, 0.4377644856770833, 0.4377644856770833, 0.46697998046875, 0.46697998046875, 0.5098673502604166, 0.5515543619791666, 0.5515543619791666, 0.5850423177083334, 0.6240234375, 0.6240234375, 0.6658121744791666, 0.7169189453125, 0.7169189453125, 0.76593017578125, 0.76593017578125, 0.8315633138020834, 0.8692220052083334, 0.8692220052083334, 0.91021728515625, 0.9471638997395834, 0.9471638997395834, 0.987548828125, 0.987548828125]
#
# webp_codec.psnr = [27.08076259438552, 28.183520253281117, 28.50240805394492, 28.78107572869167, 28.971416855520783, 29.177675953956566, 29.325828292278047, 29.479311671716154, 29.594762102428895, 29.736146093320663, 29.84070606923386, 29.97749990366106, 30.118873267192914, 30.223085465971664, 30.32422295634726, 30.405919749012998, 30.529350723023995, 30.67531666858531, 30.788097126842832, 30.831068882659665, 30.975700808977663, 31.048836672399382, 31.11871871159615, 31.245302357195115, 31.351365662472116, 31.414839622977425, 31.519025641890654, 31.62318650569997, 31.716156762681905, 31.78490410299372, 31.85720636199667, 31.95146017075974, 32.036646444629405, 32.12103471768089, 32.26736391533897, 32.31839907982114, 32.371217059986876, 32.46262598685454, 32.58720837820641, 32.707083103747124, 32.7997109713714, 32.89435933752684, 32.96336950780436, 33.02991321611319, 33.11719786802235, 33.22877468439857, 33.22877468439857, 33.3503662860035, 33.39334620216812, 33.446731553883986, 33.5189950221, 33.61712997928455, 33.665427339945346, 33.7033341667443, 33.791237526914315, 33.90270213141126, 33.90950292550946, 34.027934161913166, 34.048947134667095, 34.122184886528984, 34.19833081517354, 34.21176047457578, 34.23051035313787, 34.390793088502384, 34.44946532845282, 34.54238833724557, 34.60247217847268, 34.70841495718736, 34.70841495718736, 34.825248224315565, 34.84220957518038, 34.94401745835282, 35.017883950076424, 35.14017896732847, 35.25677375058089, 35.22629951220519, 35.51591372047217, 35.73591981353397, 36.032271063550866, 36.243424449997654, 36.47688702720297, 36.65143167070975, 36.9717419504962, 37.398432036812366, 37.681331565863076, 37.9239590413342, 38.41017438022378, 38.69459626891499, 39.04556382833309, 39.3422058167243, 39.802921941874786, 40.23290440668787, 40.687951654751814, 41.13887900032794, 41.6288655764334, 42.11128805016216, 42.53856145270555, 42.85662967484474, 43.141528187332625, 43.34467448298574]
# webp_codec.msssim = [0.8417247533798218, 0.8765644431114197, 0.8863997459411621, 0.8927357196807861, 0.8973551392555237, 0.9019343852996826, 0.9056223630905151, 0.9085562229156494, 0.9108963012695312, 0.9145863652229309, 0.9161328077316284, 0.9192385077476501, 0.9223018288612366, 0.9244083762168884, 0.9259084463119507, 0.927554726600647, 0.930051863193512, 0.9322548508644104, 0.9341764450073242, 0.936116099357605, 0.937255859375, 0.9379183053970337, 0.9389411807060242, 0.9418869018554688, 0.9434375166893005, 0.9442849159240723, 0.9455093145370483, 0.9467553496360779, 0.948455274105072, 0.9492392539978027, 0.9498960375785828, 0.95079505443573, 0.9518604874610901, 0.9527606964111328, 0.9545615315437317, 0.954813539981842, 0.9552862644195557, 0.9563899636268616, 0.9577112197875977, 0.9586931467056274, 0.9593712091445923, 0.9603307247161865, 0.9611077904701233, 0.961367666721344, 0.9619875550270081, 0.9631332755088806, 0.9631332755088806, 0.9643455147743225, 0.9643628597259521, 0.965001106262207, 0.9654661417007446, 0.9664790630340576, 0.9668293595314026, 0.9671604037284851, 0.9676307439804077, 0.9686378836631775, 0.9686169624328613, 0.9694644808769226, 0.969658613204956, 0.9703895449638367, 0.9705886840820312, 0.9706087708473206, 0.9706441760063171, 0.9724680781364441, 0.9723237752914429, 0.9728816151618958, 0.9733383655548096, 0.9744614362716675, 0.9744614362716675, 0.974494218826294, 0.9746163487434387, 0.9751322865486145, 0.9753124713897705, 0.9763570427894592, 0.9774901866912842, 0.9769828915596008, 0.9781983494758606, 0.979106068611145, 0.9807155728340149, 0.9819409251213074, 0.9820606708526611, 0.9828107357025146, 0.9840902090072632, 0.9863657355308533, 0.9867185354232788, 0.9874370098114014, 0.9890801310539246, 0.9896956086158752, 0.9904282093048096, 0.9910802841186523, 0.9919323325157166, 0.992662250995636, 0.9934847950935364, 0.9940987229347229, 0.9946237206459045, 0.9952803254127502, 0.9958130121231079, 0.996030330657959, 0.9962471127510071, 0.9964703321456909]
# webp_codec.bpp = [0.076416015625, 0.10758463541666667, 0.11962890625, 0.13077799479166666, 0.13960774739583334, 0.1468505859375, 0.15751139322916666, 0.1639404296875, 0.17093912760416666, 0.17818196614583334, 0.1859130859375, 0.19217936197916666, 0.20271809895833334, 0.2103271484375, 0.21809895833333334, 0.223876953125, 0.23124186197916666, 0.24336751302083334, 0.24983723958333334, 0.2567545572916667, 0.2642822265625, 0.2710774739583333, 0.2776692708333333, 0.2865804036458333, 0.29443359375, 0.3007405598958333, 0.3094075520833333, 0.3206787109375, 0.3299560546875, 0.338134765625, 0.3433024088541667, 0.35205078125, 0.3605143229166667, 0.3644612630208333, 0.3801676432291667, 0.3827718098958333, 0.3854573567708333, 0.3982340494791667, 0.4098714192708333, 0.4220784505208333, 0.4259033203125, 0.4394938151041667, 0.4449869791666667, 0.4505615234375, 0.46484375, 0.4717610677083333, 0.4717610677083333, 0.4846598307291667, 0.4874674479166667, 0.4968668619791667, 0.5087890625, 0.5177001953125, 0.527587890625, 0.5315755208333334, 0.5398356119791666, 0.559326171875, 0.5586344401041666, 0.5702311197916666, 0.5735270182291666, 0.593017578125, 0.5982259114583334, 0.6002197265625, 0.6020100911458334, 0.6158447265625, 0.6346435546875, 0.6508382161458334, 0.662841796875, 0.6673583984375, 0.6673583984375, 0.6910400390625, 0.6913248697916666, 0.713623046875, 0.7220458984375, 0.7357991536458334, 0.7478434244791666, 0.7475179036458334, 0.7991536458333334, 0.833251953125, 0.8729654947916666, 0.9196370442708334, 0.9605305989583334, 1.0104573567708333, 1.0498453776041667, 1.1268717447916667, 1.1842854817708333, 1.2489827473958333, 1.3658447265625, 1.4254557291666667, 1.5428873697916667, 1.6212972005208333, 1.7292073567708333, 1.8846028645833333, 2.0518798828125, 2.2331136067708335, 2.390869140625, 2.6780192057291665, 2.877685546875, 3.0206298828125, 3.3525390625, 3.534912109375]
#

artifacts_dir = ARTIFACTS_PATH / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
artifacts_dir.mkdir(parents=True, exist_ok=True)

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.figure(figsize=(900*px, 600*px))
for codec in codecs:
    if isinstance(codec, PILCodec):
        codec.eval_codec(pil_test_dataset, out_dir=artifacts_dir)
    if isinstance(codec, SwinLicCodec):
        codec.eval_codec(swin_test_dataset, out_dir=artifacts_dir)
    plt.plot(codec.bpp, codec.psnr, label=codec.label, marker='o')

plt.grid(True)
plt.legend()
plt.xlim(0, 1.2)
plt.ylim(20, 40)
plt.yticks(range(20, 40, 2))
plt.xlabel('Szybkość bitowa [bpp]')
plt.ylabel('PSNR [dB]')
plt.margins(0, 0)
plt.tight_layout()
plt.savefig(artifacts_dir / "rd_psnr.png")

plt.figure(figsize=(900*px, 600*px))
for codec in codecs:
    print(f"Codec: {codec.label}, PSNR: {codec.psnr}, MS-SSIM: {codec.msssim}, bpp: {codec.bpp}")
    plt.plot(codec.bpp, codec.msssim, label=codec.label, marker='o')

plt.grid(True)
plt.legend()
plt.xlim(0, 1.2)
plt.xlabel('Szybkość bitowa [bpp]')
plt.ylabel('MS-SSIM')
plt.margins(0, 0)
plt.tight_layout()
plt.savefig(artifacts_dir / "rd_ms_ssim.png")

# print(f"AVIF: 0,3: {avif_codec.bpp[43]}, 0,6: {avif_codec.bpp[56]}, 1,0: {avif_codec.bpp[65]}")
# print(f"SwinLIC: 0,3: {swin_lic_codec.bpp[4]}, 0,6: {swin_lic_codec.bpp[2]}, 1,0: {swin_lic_codec.bpp[0]}")
# print(f"WebP: 0,3: {webp_codec.bpp[30]}, 0,6: {webp_codec.bpp[64]}, 1,0: {webp_codec.bpp[79]}")
# print(f"JPEG: 0,3: {jpeg_codec.bpp[7]}, 0,6: {jpeg_codec.bpp[18]}, 1,0: {jpeg_codec.bpp[31]}")