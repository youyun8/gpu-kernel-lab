# 06-pytorch-integration — PyTorch 實戰

對應網站 Track D (第 17–21 章)。這裡的範例需要安裝 PyTorch (CUDA 或 ROCm 版)。無 GPU 時仍可閱讀程式碼並做語法檢查。

## 目錄

- `profile_model.py` — 第 17 章:用 `torch.profiler` 找出模型中最耗時的 op 並匯出 trace。
- `cpp_extension/` — 第 18 章:完整的 GELU forward+backward extension (`.cpp` + `.cu` + `setup.py` + `test_gelu.py`),含 `gradcheck`。
- `load_inline/` — 第 19 章:用 `load_inline` 的 fused bias+gelu+rowsum,對照 baseline 與 `torch.compile`。
- `triton/` — 第 20 章:Triton 版 softmax 與 matmul,對照 PyTorch reference。
- `custom_op_autograd/` — 第 21 章:用 `torch.library.custom_op` 註冊、`torch.compile` 相容性、regression benchmark。

## 執行

```bash
pip install torch            # CUDA 或 ROCm 對應版本
pip install triton           # 第 20 章需要
python profile_model.py
python cpp_extension/test_gelu.py
python load_inline/fused_bias_gelu.py
python triton/softmax_triton.py
python custom_op_autograd/test_custom_op.py
```

## 驗證狀態

這些是 Python 範例,以 `python -m py_compile` 做語法驗證(見 `VERIFICATION_LOG.md`)。實際執行需要對應的 GPU + PyTorch 環境。
