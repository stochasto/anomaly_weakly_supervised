import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")


def test(net, config, test_loader, test_info, step, model_file=None):
    """
    Test loop compatible with [T,10,2048] features.
    Handles I3D 10-crop averaging and GT truncation mismatch.
    """
    with torch.no_grad():
        net.eval()
        net.flag = "Test"

        # Load checkpoint if provided
        if model_file is not None:
            print(f"[INFO] Loading checkpoint from: {model_file}")
            net.load_state_dict(torch.load(model_file))

        # Ground truth (frame-level)
        frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None

        cls_label, cls_pre = [], []
        temp_predict = torch.zeros((0)).cuda()
        count = 0

        load_iter = iter(test_loader)

        for i in range(len(test_loader.dataset)):
            # Loader expected to yield (features, label, name)
            _data, _label, _name = next(load_iter)
            _data = _data.cuda(non_blocking=True)
            _label = _label.cuda(non_blocking=True)

            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            # Aggregate 10-crop predictions per video
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()  # mean over crops

                # Binary decision (simple threshold on max frame score)
                cls_pre.append(1 if a_predict.max() > 0.5 else 0)

                # =========================
                # Robust frame-level expansion
                # =========================
                num_frames_remaining = len(frame_gt) - (0 if frame_predict is None else len(frame_predict))
                num_frames_clip = len(a_predict)

                # Repeat predictions to roughly match remaining frames
                repeat_factor = max(num_frames_remaining // num_frames_clip, 1)
                fpre_ = np.repeat(a_predict, repeat_factor)

                # Trim if it exceeds remaining GT frames
                fpre_ = fpre_[:num_frames_remaining]
                

                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])

                temp_predict = torch.zeros((0)).cuda()  # reset
                count += len(fpre_)

        # ==========================
        # Compute metrics
        # ==========================
        print(f'len gt {len(frame_gt)} len pred {len(frame_predict)}')
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)

        correct_num = np.sum(np.array(cls_label) == np.array(cls_pre))
        accuracy = correct_num / len(cls_pre) if len(cls_pre) > 0 else 0.0

        # ==========================
        # Update test info
        # ==========================
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)

        print(f"[STEP {step:04d}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={accuracy:.4f}")

        return auc_score, ap_score, accuracy
