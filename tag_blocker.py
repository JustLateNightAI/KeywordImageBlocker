import os
import torch
from PIL import Image
from torchvision import transforms

class TagKeywordBlocker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "tags":     ("STRING", {"multiline": True, "default": "Plug a image tagger like WD14 into this box"}),
                "keywords": ("STRING", {"multiline": True, "default": "List the tags you want to filter for and block in this text box, ie nude, penis, naked, ect"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "filter_image"
    CATEGORY     = "Keyword Image Blocker"

    def __init__(self):
        base = os.path.dirname(__file__)
        self.warning_path = os.path.join(base, "assets", "warning.png")
        self.to_tensor    = transforms.ToTensor()

    def _to_hwc(self, t: torch.Tensor) -> torch.Tensor:
        """Convert any [1,C,H,W], [C,H,W] or [1,H,W,C], [H,W,C] → [H,W,3]"""
        print(f"[DEBUG _to_hwc] before: dim={t.dim()}, shape={tuple(t.shape)}")
        # drop batch dim
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
            print(f"[DEBUG _to_hwc] dropped batch → shape={tuple(t.shape)}")

        # now t.dim()==3
        if t.dim() != 3:
            raise RuntimeError(f"[DEBUG _to_hwc] Unexpected tensor dim: {t.shape}")

        # channels-first [C,H,W]?
        if t.shape[0] == 3:
            t = t.permute(1, 2, 0)  # → [H,W,C]
            print(f"[DEBUG _to_hwc] permuted CHW→HWC → shape={tuple(t.shape)}")

        # or maybe it's already [H,W,C]?
        elif t.shape[-1] == 3:
            print(f"[DEBUG _to_hwc] already HWC → shape={tuple(t.shape)}")
        else:
            raise RuntimeError(f"[DEBUG _to_hwc] No channel dim found in: {t.shape}")

        return t

    def filter_image(self, image, tags, keywords):
        # break inputs into list
        frames = image if isinstance(image, (list, tuple)) else [image]

        # log incoming frames
        print(f"[DEBUG filter_image] got {len(frames)} frame(s)")
        for i, frm in enumerate(frames):
            if isinstance(frm, torch.Tensor):
                print(f"  Frame {i}: Tensor dim={frm.dim()}, shape={tuple(frm.shape)}, "
                      f"dtype={frm.dtype}, device={frm.device}")
            else:
                print(f"  Frame {i}: {type(frm)} size={frm.size if hasattr(frm, 'size') else 'N/A'}")

        # keyword check
        raw = str(tags).lower()
        kws = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        hit = False

        for keyword in kws:
            groups = keyword.split("+")
            all_groups_met = True
            
            for group in groups:
                alternatives = group.split("/")
                found_in_group = False
                
                for alt in alternatives:
                    if alt in raw:
                        found_in_group = True
                        break
                        
                if not found_in_group:
                    all_groups_met = False
                    break
                    
            if all_groups_met:
                hit = True
                break

        print(f"[DEBUG filter_image] tags='{raw}', keywords={kws}, hit={hit}")

        # pick largest tensor frame to size warning
        best, area = None, -1
        for frm in frames:
            if not isinstance(frm, torch.Tensor):
                continue
            t = frm
            # detect H×W
            if t.dim() == 4 and t.shape[-1] == 3:
                H, W = t.shape[1], t.shape[2]
            elif t.dim() == 4 and t.shape[1] == 3:
                H, W = t.shape[2], t.shape[3]
            elif t.dim() == 3 and t.shape[0] == 3:
                H, W = t.shape[1], t.shape[2]
            elif t.dim() == 3 and t.shape[-1] == 3:
                H, W = t.shape[0], t.shape[1]
            else:
                continue
            a = H * W
            if a > area:
                best, area = frm, a

        # if no tensor frames, bail out and just pass originals in HWC
        if best is None or not hit:
            out = []
            for frm in frames:
                if isinstance(frm, torch.Tensor):
                    out.append(self._to_hwc(frm))
                else:
                    # fallback: convert PIL→tensor HWC
                    t = self.to_tensor(frm).permute(1, 2, 0)
                    out.append(t)
            print("[DEBUG filter_image] no hit or no tensors → passing through HWC tensors")
            return (out,)

        # on hit: size warning to best frame
        # extract H,W from best
        t0 = best
        if t0.dim() == 4 and t0.shape[-1] == 3:
            H, W = t0.shape[1], t0.shape[2]
        elif t0.dim() == 4 and t0.shape[1] == 3:
            H, W = t0.shape[2], t0.shape[3]
        elif t0.dim() == 3 and t0.shape[0] == 3:
            H, W = t0.shape[1], t0.shape[2]
        else:
            H, W = t0.shape[0], t0.shape[1]

        # load & resize warning
        pil_warn = Image.open(self.warning_path).convert("RGB")
        pil_warn = pil_warn.resize((W, H), resample=Image.LANCZOS)

        # to tensor CHW→HWC
        warn_t = self.to_tensor(pil_warn).permute(1, 2, 0)
        warn_t = warn_t.to(best.device).to(best.dtype)
        print(f"[DEBUG filter_image] hit → warning tensor shape={tuple(warn_t.shape)}")

        # return single-element list
        return ([warn_t],)




NODE_CLASS_MAPPINGS = {
    "TagKeywordBlocker": TagKeywordBlocker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TagKeywordBlocker": "Keyword Image Blocker"
}
