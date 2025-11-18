import os
import typing

import PIL.Image
import compressai
import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
from torch.nn.parameter import Parameter
from torchvision.transforms.v2.functional import to_pil_image

from src.data.coco_dataset import CocoDataset
from src.data.imagenet_dataset import ImageNetDataset
from src.data.kodak_dataset import KodakDataset
from src.data.transforms import YCbCrCompression, YCbCrDecompression, RGBDecompression, RGBCompression, denormalize, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.checkpoint_helpers import load_training_state_with_clearml_from_file
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config

def set_entropy_coder_precision(module, precision=16):
    """
    Rekurencyjnie przeszukuje moduł i ustawia entropy_coder_precision
    we wszystkich podmodułach, które mają ten atrybut.
    """
    # Sprawdź czy obecny moduł ma atrybut entropy_coder_precision
    if hasattr(module, 'entropy_coder_precision'):
        print(f"Ustawiam precision={precision} w {module.__class__.__name__}")
        setattr(module, 'entropy_coder_precision', precision)

    # Rekurencyjnie przeszukaj wszystkie podmoduły
    for name, child in module.named_children():
        print(f"Sprawdzam podmoduł: {name}")
        set_entropy_coder_precision(child, precision)


def diagnose_model_state(model):
    """
    Diagnoza stanu modelu po update
    """
    print("=== DIAGNOZA STANU MODELU ===")

    for name, module in model.named_modules():
        if hasattr(module, '_quantized_cdf'):
            cdf_shape = module._quantized_cdf.shape
            offset_shape = module._offset.shape
            cdf_length_shape = module._cdf_length.shape

            print(f"\n{name}:")
            print(f"  _quantized_cdf: {cdf_shape}")
            print(f"  _offset: {offset_shape}")
            print(f"  _cdf_length: {cdf_length_shape}")

            # Sprawdź czy bufory są puste
            if module._quantized_cdf.numel() == 0:
                print(f"  ❌ BŁĄD: {name} ma pusty _quantized_cdf!")
            else:
                print(f"  ✅ {name} ma prawidłowe bufory")

            # Sprawdź zakres wartości
            if module._quantized_cdf.numel() > 0:
                cdf_min = module._quantized_cdf.min().item()
                cdf_max = module._quantized_cdf.max().item()
                print(f"  CDF range: [{cdf_min}, {cdf_max}]")

                if cdf_max > 1000000:  # Podejrzane wartości
                    print(f"  ⚠️  OSTRZEŻENIE: Bardzo wysokie wartości CDF!")


def check_entropy_parameters(model):
    """
    Sprawdź parametry modułów entropy
    """
    print("=== SPRAWDZANIE PARAMETRÓW ENTROPY ===")

    # Sprawdź GaussianConditional
    gaussian_module = model.latent_codec.y.y.gaussian_conditional
    print(f"\n🔍 GaussianConditional:")
    print(f"  scale_table shape: {gaussian_module.scale_table.shape}")
    print(f"  scale_table range: [{gaussian_module.scale_table.min():.6f}, {gaussian_module.scale_table.max():.6f}]")

    # Sprawdź czy scale_table ma prawidłowe wartości
    if gaussian_module.scale_table.min() <= 0:
        print("  ❌ BŁĄD: scale_table zawiera wartości <= 0!")
    else:
        print("  ✅ scale_table ma prawidłowe wartości")

    # Sprawdź EntropyBottleneck
    entropy_module = model.latent_codec.hyper.entropy_bottleneck
    print(f"\n🔍 EntropyBottleneck:")

    # Sprawdź parametry entropy bottleneck
    if hasattr(entropy_module, '_matrices'):
        print(f"  _matrices shape: {entropy_module._matrices.shape}")
        print(f"  _matrices range: [{entropy_module._matrices.min():.6f}, {entropy_module._matrices.max():.6f}]")
    if hasattr(entropy_module, '_biases'):
        print(f"  _biases shape: {entropy_module._biases.shape}")
        print(f"  _biases range: [{entropy_module._biases.min():.6f}, {entropy_module._biases.max():.6f}]")
    if hasattr(entropy_module, '_factors'):
        print(f"  _factors shape: {entropy_module._factors.shape}")
        print(f"  _factors range: [{entropy_module._factors.min():.6f}, {entropy_module._factors.max():.6f}]")


def check_model_weights(model):
    """
    Sprawdź czy wagi modelu są prawidłowe
    """
    print("=== SPRAWDZANIE WAG MODELU ===")

    problem_found = False

    for name, param in model.named_parameters():
        if 'entropy' in name.lower() or 'gaussian' in name.lower():
            print(f"\n📊 {name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Range: [{param.min():.6f}, {param.max():.6f}]")
            print(f"  Mean: {param.mean():.6f}")
            print(f"  Std: {param.std():.6f}")

            # Sprawdź problematyczne wartości
            if torch.isnan(param).any():
                print(f"  ❌ BŁĄD: {name} zawiera NaN!")
                problem_found = True
            elif torch.isinf(param).any():
                print(f"  ❌ BŁĄD: {name} zawiera Inf!")
                problem_found = True
            elif param.std() == 0:
                print(f"  ⚠️  OSTRZEŻENIE: {name} ma zerowe odchylenie standardowe!")
            else:
                print(f"  ✅ {name} wygląda poprawnie")

    return not problem_found


def manually_build_buffers(model):
    """
    Manualnie zbuduj bufory entropy
    """
    print("=== MANUALNE BUDOWANIE BUFORÓW ===")

    # 1. Napraw GaussianConditional
    gaussian_module = model.latent_codec.y.y.gaussian_conditional
    print("\n🔧 Naprawiam GaussianConditional...")

    try:
        # Sprawdź czy scale_table jest prawidłowa
        if gaussian_module.scale_table.min() <= 0:
            print("  Naprawiam scale_table...")
            # Zastąp ujemne/zerowe wartości małą dodatnią wartością
            gaussian_module.scale_table.data = torch.clamp(gaussian_module.scale_table.data, min=1e-6)

        # Manualnie wywołaj update
        gaussian_module.update_scale_table(gaussian_module.scale_table, force=True)
        print("  ✅ GaussianConditional naprawiony")

    except Exception as e:
        print(f"  ❌ Błąd naprawy GaussianConditional: {e}")

    # 2. Sprawdź stan po naprawie
    print(f"  Nowy stan _quantized_cdf: {gaussian_module._quantized_cdf.shape}")

    # 3. Napraw EntropyBottleneck
    entropy_module = model.latent_codec.hyper.entropy_bottleneck
    print("\n🔧 Naprawiam EntropyBottleneck...")

    try:
        # Sprawdź czy parametry są prawidłowe
        if hasattr(entropy_module, '_matrices'):
            if torch.isnan(entropy_module._matrices).any() or torch.isinf(entropy_module._matrices).any():
                print("  ❌ _matrices zawiera NaN lub Inf!")
                return False

        # Manualnie wywołaj update
        entropy_module.update()
        print("  ✅ EntropyBottleneck naprawiony")

    except Exception as e:
        print(f"  ❌ Błąd naprawy EntropyBottleneck: {e}")
        return False

    # 4. Sprawdź stan po naprawie
    print(f"  Nowy stan _quantized_cdf: {entropy_module._quantized_cdf.shape}")

    return True


def test_compression_pipeline(model):
    """
    Przetestuj pełny pipeline kompresji
    """
    print("=== TEST PIPELINE KOMPRESJI ===")

    # Test z małym obrazem
    test_sizes = [
        (64, 64),
        (128, 128),
        (256, 256)
    ]

    for h, w in test_sizes:
        print(f"\n🧪 Test z obrazem {h}x{w}")
        test_tensor = torch.randn(1, 3, h, w)

        try:
            # 1. Forward pass
            with torch.no_grad():
                print("  Forward pass...")
                output = model(test_tensor)
                print(f"  ✅ Forward OK - output shape: {output['x_hat'].shape}")

                # 2. Compress
                print("  Compress...")
                compressed = model.compress(test_tensor)
                print(f"  ✅ Compress OK - strings: {len(compressed['strings'])}")

                # 3. Decompress
                print("  Decompress...")
                decompressed = model.decompress(compressed['strings'], compressed['shape'])
                print(f"  ✅ Decompress OK - shape: {decompressed['x_hat'].shape}")

                # 4. Sprawdź jakość
                mse = torch.nn.functional.mse_loss(test_tensor, decompressed['x_hat'])
                print(f"  📊 MSE: {mse.item():.6f}")

        except Exception as e:
            print(f"  ❌ Błąd dla {h}x{w}: {e}")
            return False

    return True


def verify_cdf_values(model):
    """
    Sprawdź czy wartości CDF są rozsądne
    """
    print("=== WERYFIKACJA WARTOŚCI CDF ===")

    # GaussianConditional
    gaussian_module = model.latent_codec.y.y.gaussian_conditional
    cdf = gaussian_module._quantized_cdf
    offset = gaussian_module._offset
    cdf_length = gaussian_module._cdf_length

    print(f"\n📊 GaussianConditional CDF:")
    print(f"  Shape: {cdf.shape}")
    print(f"  Range: [{cdf.min()}, {cdf.max()}]")
    print(f"  Offset range: [{offset.min()}, {offset.max()}]")
    print(f"  CDF length range: [{cdf_length.min()}, {cdf_length.max()}]")

    # Sprawdź czy wartości są rozsądne (powinna być monotonicznie rosnąca)
    for i in range(min(5, cdf.shape[0])):  # Sprawdź pierwsze 5 kanałów
        row = cdf[i]
        if not torch.all(row[1:] >= row[:-1]):
            print(f"  ⚠️  OSTRZEŻENIE: CDF[{i}] nie jest monotonicznie rosnąca!")

    # EntropyBottleneck
    entropy_module = model.latent_codec.hyper.entropy_bottleneck
    cdf = entropy_module._quantized_cdf
    offset = entropy_module._offset
    cdf_length = entropy_module._cdf_length

    print(f"\n📊 EntropyBottleneck CDF:")
    print(f"  Shape: {cdf.shape}")
    print(f"  Range: [{cdf.min()}, {cdf.max()}]")
    print(f"  Offset range: [{offset.min()}, {offset.max()}]")
    print(f"  CDF length range: [{cdf_length.min()}, {cdf_length.max()}]")

    # Sprawdź monotoniczność
    for i in range(min(5, cdf.shape[0])):
        row = cdf[i]
        if not torch.all(row[1:] >= row[:-1]):
            print(f"  ⚠️  OSTRZEŻENIE: CDF[{i}] nie jest monotonicznie rosnąca!")


def test_symbol_values(model):
    """
    Test czy symbole mają rozsądne wartości
    """
    print("=== TEST WARTOŚCI SYMBOLI ===")

    test_tensor = torch.randn(1, 3, 128, 128)

    try:
        with torch.no_grad():
            # Przechwytuj wartości w procesie kompresji
            print("Rozpoczynam kompresję z monitorowaniem...")

            # Forward pass do uzyskania latent representations
            y = model.g_a(test_tensor)
            z = model.h_a(y)

            print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
            print(f"z range: [{z.min():.3f}, {z.max():.3f}]")

            # Sprawdź kwantyzację
            z_hat, z_likelihoods = model.latent_codec.hyper.entropy_bottleneck(z)
            print(f"z_hat range: [{z_hat.min():.3f}, {z_hat.max():.3f}]")

            # Sprawdź czy symbole są rozsądne
            z_symbols = model.latent_codec.hyper.entropy_bottleneck.quantize(z, "symbols")
            print(f"z_symbols range: [{z_symbols.min()}, {z_symbols.max()}]")

            if z_symbols.max() > 100000:  # Sprawdź czy nadal występuje problem
                print("❌ PROBLEM: Symbole nadal mają bardzo wysokie wartości!")
                return False
            else:
                print("✅ Symbole mają rozsądne wartości")

    except Exception as e:
        print(f"❌ Błąd podczas testu: {e}")
        return False

    return True


def analyze_forward_pass(model):
    """
    Analizuj forward pass modelu
    """
    print("=== ANALIZA FORWARD PASS ===")

    test_tensor = torch.randn(1, 3, 64, 64)

    try:
        # Sprawdź czy model ma standardowy forward
        print("🧪 Testuję standardowy forward...")
        with torch.no_grad():
            output = model(test_tensor)
            print(f"✅ Forward pass OK!")
            print(f"   Output type: {type(output)}")

            if isinstance(output, dict):
                print("   Output keys:", list(output.keys()))
                for key, value in output.items():
                    if hasattr(value, 'shape'):
                        print(f"   {key}: {value.shape}")
                    else:
                        print(f"   {key}: {type(value)}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

    # Sprawdź metodę compress
    try:
        print("\n🧪 Testuję compress bezpośrednio...")
        with torch.no_grad():
            compressed = model.compress(test_tensor)
            print(f"✅ Compress OK!")
            print(f"   Compressed type: {type(compressed)}")

    except Exception as e:
        print(f"❌ Compress failed: {e}")



# Test
if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # device = "cpu"
    print("Device:", device)
    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    compressai.set_entropy_coder('ans')

    models = [
        # ("SWIN-S-IC_0.20.3", "gdn_swin_v2_s"),
        # ("SWIN-S-IC-BASE_0.58.0", "gdn_swin_v2_s"),
        ("SWIN-B-IC_0.7.1.5", "gdn_swin_v2_b"),
        # ("SWIN-S-IC_0.3.1_100", "swin_v2_s")
        # "SWIN-T-IC_0.12.0-150of400",
        # "SWIN-T-IC_0.9.4-210of400"
    ]

    transform = RGBCompression(noresize=True, normalize=False, mean=(0.470, 0.447, 0.408), std=(0.270, 0.266, 0.281))
    target_transform = RGBCompression(normalize=False, noresize=True, mean=(0.470, 0.447, 0.408), std=(
        0.270, 0.266, 0.281))
    # transform = RGBCompression(crop_size=256, resize_size=256)

    dataset = KodakDataset(transform=transform, target_transform=target_transform)


    pic_num = 20

    for m in models:
        psnr_sum = 0
        bpp_sum = 0
        ssim_sum = 0
        dataloader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False))
        model: SwinTransformerCompressionAutoencoder = typing.cast(SwinTransformerCompressionAutoencoder, model_from_config(
            {
                "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
                # "weights": "SWIN-S-IC_0.70.10",
                "args": {
                    "pretrained_encoder": False,
                    "encoder_type": "gdn_swin_v2_b",
                    "encoder_embed_dim": 128,
                    "encoder_dims": [128, 256, 512, 1024],
                    "encoder_depths": [2, 2, 8, 4],
                    "encoder_num_heads": [4, 8, 16, 32],
                    "encoder_window_size": [8, 8],
                    "encoder_sd_factor": 0.1,
                    "encoder_mlp_ratio": 3,
                    "decoder_depths": [4, 8, 2, 2],
                    "decoder_dims": [1024, 512, 256, 128, 64],
                    "decoder_num_heads": [32, 16, 8, 4],
                    "decoder_window_size": [[8, 8], [8, 8], [8, 8], [8, 8]],
                    "decoder_mlp_ratio": [3, 3, 3, 3],
                    "decoder_sd_factor": 0.05,
                    "no_compress": False,
                    "checkpointing": False,
                }
            }))
        # state = torch.load("D:\\Studia\\INZ\\checkpoint\\last_checkpoint.pth", map_location='cpu')
        state = torch.load("/run/media/jakub/Dane/Studia/INZ/checkpoint/checkpoint_130000.pth", map_location='cpu')
        model.load_state_dict(state['model'])
        print(state['model'].keys())
        model.eval()

        # diagnose_model_state(model)
        # check_entropy_parameters(model)

        # Sprawdź wagi
        # weights_ok = check_model_weights(model)
        model.update(force=True, update_quantiles=True)
        # model.update()
        # print(weights_ok)
        # manually_build_buffers(model)
        # model.update(force=True, update_quantiles=True)
        model.to(device)
        # test_compression_pipeline(model)
        # verify_cdf_values(model)
        # test_symbol_values(model)
        # analyze_forward_pass(model)
        params: typing.Iterator[Parameter]  = model.g_s.reconstruction.activation.parameters()
        for param in params:
            print(param.data)
        torch.save(model.state_dict(), f"../../models/{m[0]}.pth")
        print(sum(param.numel() for param in model.parameters() if param.requires_grad))
        for iteration in range(pic_num):
            x_batch, _ = next(dataloader_iter)
            x_batch = x_batch.to(device)

            with torch.no_grad():
                compress_output = model.compress(x_batch)
                b_repr, shape = compress_output['strings'], compress_output['shape']
                x_recon = model.decompress(b_repr, shape)['x_hat']
                # output = model(x_batch)
                # x_recon, y_likelihoods = output['x_hat'], None

            output_transform = RGBDecompression(denorm=False, mean=(0.470, 0.447, 0.408), std=(0.270, 0.266,
                                                                                              0.281)).to(x_batch.device)

            in_img: PIL.Image.Image = output_transform(x_batch)[0] # TODO: Examine
            out_img: PIL.Image.Image = to_pil_image(x_recon[0])
            out_img = out_img.crop((0, 0, in_img.size[0], in_img.size[1]))

            os.makedirs(os.path.join(ARTIFACTS_PATH, m[0]), exist_ok=True)
            im_img_path = os.path.join(ARTIFACTS_PATH, m[0], f"{iteration}_original.png")
            out_img_path = os.path.join(ARTIFACTS_PATH, m[0], f"{iteration}_compressed.png")
            in_img.save(im_img_path)
            out_img.save(out_img_path)

            x_orig_np = denormalize(x_batch, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)[0].cpu().numpy()
            x_recon_np = x_recon[0].cpu().numpy()

            image1 = imread(im_img_path)
            image2 = imread(out_img_path)

            psnr_value = peak_signal_noise_ratio(image1, image2, data_range=255.0)
            print(f"PSNR: {psnr_value}")
            psnr_sum += psnr_value

            # ssim_value = structural_similarity(image1, image2, min_size=7)
            # print(f"ssim: {ssim_value}")
            # ssim_sum += ssim_value
            bits = sum([sum([len(b) for b in b_repr_item]) * 8 / len(b_repr_item) for b_repr_item in b_repr])
            bpp = bits / (in_img.size[0] * in_img.size[1])
            print(f"bpp: {bpp}")
            bpp_sum += bpp
            torch.cuda.empty_cache()

        print(f"Avg PSNR: {(psnr_sum / pic_num):.4f}")
        print(f"Avg SSIM: {(ssim_sum / pic_num):.4f}")
        print(f"Avg bpp: {(bpp_sum / pic_num):.4f}")


