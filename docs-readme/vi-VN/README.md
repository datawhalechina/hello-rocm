<div align=center>
  <img src="../../docs/public/images/head.png" >
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*Mã nguồn mở · Cộng đồng vận hành · Giúp Hệ sinh thái AI của AMD dễ tiếp cận hơn*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>
<div align="center">

<a href="https://datawhalechina.github.io/hello-rocm/"><img src="https://img.shields.io/badge/hello--rocm_Hướng_dẫn_đầy_đủ-Trải_nghiệm-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="25"></a>

</div>

&emsp;&emsp;Kể từ **ROCm 7.10.0** (phát hành ngày 11 tháng 12 năm 2025), ROCm có thể được cài đặt liền mạch trong môi trường ảo Python giống như CUDA, với hỗ trợ chính thức cho cả **Linux và Windows**. Đây là một bước tiến lớn của AMD trong lĩnh vực AI: người học và những người đam mê LLM không còn bị giới hạn bởi phần cứng NVIDIA—GPU AMD là một lựa chọn mạnh mẽ và thiết thực.

&emsp;&emsp;Tuy nhiên, việc hạ thấp rào cản phần cứng không tự động làm rõ lộ trình học tập. Đối với những người học đã có nền tảng LLM và muốn đưa chúng vào thực hành trên GPU AMD, những thách thức thực sự mới chỉ bắt đầu: Làm thế nào để triển khai một mô hình trên GPU AMD? Làm thế nào để tinh chỉnh và huấn luyện trên đó? Làm thế nào để hiểu hệ thống lập trình GPU của ROCm và hoàn thành việc di chuyển từ CUDA sang ROCm? Và cuối cùng, làm thế nào để tập hợp tất cả những khả năng này thành một ứng dụng AI thực sự, hoạt động được?

&emsp;&emsp;**hello-rocm** được sinh ra chính xác cho con đường này. Dự án này bao phủ một cách có hệ thống chuỗi sử dụng hoàn chỉnh của các mô hình lớn trên nền tảng AMD ROCm, đưa bạn từ việc chạy mô hình đầu tiên đến xây dựng các ứng dụng AI thực sự trên GPU AMD, qua mọi bước quan trọng bao gồm tinh chỉnh, huấn luyện và lập trình GPU—biến GPU AMD không chỉ là một card đồ họa, mà là điểm khởi đầu thực sự của bạn vào thế giới phát triển AI.

&emsp;&emsp;**Dự án này chủ yếu là các hướng dẫn** để sinh viên và những người thực hành trong tương lai có thể học AMD ROCm một cách có cấu trúc. **Mọi người đều được chào đón mở issue hoặc gửi pull request** để cùng nhau phát triển và duy trì dự án.

> &emsp;&emsp;***Lộ trình học tập: Hoàn thành [00-Môi trường](../../docs/en/00-environment/index.md) trước (ROCm + PyTorch + **uv**), sau đó đến triển khai và tinh chỉnh, và cuối cùng là tối ưu hóa toán tử và lập trình GPU. Sau khi môi trường của bạn hoạt động, LM Studio hoặc vLLM là một điểm khởi đầu tốt.***

### hello-rocm Skill: dùng dự án này trong trợ lý AI của bạn

&emsp;&emsp;Nếu bạn dùng công cụ lập trình AI hỗ trợ Skills, Rules hoặc cấu hình Agent, bạn có thể dùng **hello-rocm Skill** tích hợp sẵn. Skill này dựa trên cấu trúc kho, chỉ mục Reference, bảng kiến trúc GPU, hướng dẫn triển khai và danh sách khắc phục sự cố để dẫn bạn tới tài liệu và liên kết chính thức phù hợp.

```text
Use src/hello-rocm-skill in the current repository as the hello-rocm Skill. If your tool supports Skills, Rules, or Agent configuration, install or load it in the appropriate place, such as .claude/skills, .cursor/skills, or .agents/skills, then use that Skill to help me learn, deploy, and troubleshoot AMD ROCm.
```

&emsp;&emsp;Thử hỏi: GPU AMD của tôi có hỗ trợ ROCm không? Con đường nhanh nhất để chạy LLM cục bộ đầu tiên của tôi là gì? Làm thế nào để cài đặt vLLM/Ollama/llama.cpp trên ROCm? Làm thế nào để gỡ lỗi `torch.cuda.is_available()` trả về False? Xem [hướng dẫn hello-rocm Skill](../../docs/en/04-references/index.md#hello-rocm-skill).

### Cập nhật mới nhất

- *15/05/2026:* [*Ghi chú phát hành ROCm 7.13.0*](https://rocm.docs.amd.com/en/7.13.0-preview/index.html)

- *11/03/2026:* [*Ghi chú phát hành ROCm 7.12.0*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *11/12/2025:* [*Ghi chú phát hành ROCm 7.10.0*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Các mô hình & hướng dẫn được hỗ trợ

<p align="center">
  <strong>✨ Các LLM chính thống: môi trường · suy luận đa framework · tinh chỉnh ✨</strong><br>
  <em>Thiết lập ROCm thống nhất (Windows / Ubuntu) + ROCm 7+ · hướng dẫn theo từng mô hình (đang phát triển)</em><br>
 <a href="../../docs/en/00-environment/index.md">00 — Thiết lập môi trường</a>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">

  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/qwen3/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/qwen3/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/qwen3/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/qwen3/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/02-fine-tune/qwen3/qwen3-0.6b-lora-swanlab.md">Qwen3-0.6B LoRA + SwanLab</a><br>
      • <a href="../../src/fine-tune/models/qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA (Notebook)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3.5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/qwen3.5/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/qwen3.5/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/qwen3.5/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/qwen3.5/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/02-fine-tune/qwen3.5/qwen3.5-4b-lora-swanlab.md">Qwen3.5-4B LoRA + SwanLab</a><br>
      • <a href="../../src/fine-tune/models/qwen3.5/Qwen3.5-4B-LoRA.ipynb">Qwen3.5-4B LoRA (Notebook)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/gemma4/gemma4_model.md">Tổng quan mô hình Gemma 4</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../docs/en/01-deploy/gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../src/fine-tune/models/gemma4/gemma4_emotion_lora_modelscope_single_gpu.ipynb">Tinh chỉnh LoRA Gemma4 E4B (TRL, Notebook)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/minicpm/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM-V 4.6</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/minicpmv/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
      • <a href="../../docs/en/01-deploy/minicpmv/vllm-rocm7-deploy.md">vLLM</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>MiniCPM-o 4.5</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/01-deploy/minicpm-o/minicpm-o-model.md">Giới thiệu mô hình MiniCPM-o 4.5</a><br>
      • <a href="../../docs/en/01-deploy/minicpm-o/llamacpp-omni-rocm7-deploy.md">Triển khai llama.cpp-omni (giọng nói + thị giác + TTS)</a><br>
      • <a href="../../docs/en/01-deploy/minicpm-o/webdemo-rocm7-deploy.md">Triển khai Web Demo song công</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Chính sách Embodied AI (ACT / SmolVLA / Pi0 / Pi0.5)</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../docs/en/05-amd-yes/every-embodied.md">Every Embodied — Sao chép chính sách Embodied AI ROCm</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/08_act_training_rocm.ipynb">Notebook huấn luyện ACT ROCm</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/09_smolvla_training_rocm.ipynb">Notebook huấn luyện SmolVLA ROCm</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/10_pi0_training_rocm.ipynb">Notebook huấn luyện Pi0 ROCm</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/12_pi0_strict_input_end_to_end.ipynb">Notebook chẩn đoán nghiêm ngặt đầu cuối Pi0</a><br>
      • <a href="https://github.com/datawhalechina/every-embodied/blob/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/13_pi05_random_position_eef_delta.ipynb">Notebook huấn luyện Pi0.5 EEF-delta</a><br>
    </td>
  </tr>
</table>

## Tại sao lại có dự án này

&emsp;&emsp;ROCm là gì?

> ROCm (Radeon Open Compute) là ngăn xếp tính toán GPU mở của AMD dành cho HPC và máy học. Nó cho phép bạn chạy các khối lượng công việc song song trên GPU AMD và là con đường thay thế CUDA chính trên phần cứng AMD.

&emsp;&emsp;Các LLM mở có mặt ở khắp mọi nơi, nhưng hầu hết các hướng dẫn và công cụ đều giả định ngăn xếp NVIDIA CUDA. Các nhà phát triển chọn AMD thường thiếu tài liệu học tập toàn diện, thuần ROCm.
&emsp;&emsp;Từ **ROCm 7.10.0** (ngày 11 tháng 12 năm 2025), dự án **TheRock** của AMD đã tách runtime tính toán khỏi hệ điều hành, cho phép cùng một giao diện ROCm chạy trên cả **Linux và Windows**, đồng thời ROCm có thể được cài đặt vào môi trường Python tương tự như CUDA. ROCm không còn là "hệ thống ống nước chỉ dành cho Linux" nữa—mà là một nền tảng tính toán AI đa nền tảng. **hello-rocm** tập hợp các hướng dẫn thực tế để nhiều người hơn có thể thực sự sử dụng GPU AMD cho việc huấn luyện và suy luận.

&emsp;&emsp;***Chúng tôi hy vọng trở thành cầu nối giữa GPU AMD và những người xây dựng hàng ngày—cởi mở, bao trùm và hướng tới một tương lai AI rộng lớn hơn.***

## Dành cho ai

&emsp;&emsp;Bạn có thể thấy dự án này hữu ích nếu bạn:

* Có GPU AMD và muốn chạy LLM cục bộ;
* Muốn xây dựng trên nền tảng AMD nhưng thiếu một chương trình học ROCm có cấu trúc;
* Quan tâm đến việc triển khai và suy luận hiệu quả về chi phí;
* Tò mò về ROCm và thích học tập thực hành.

## Lộ trình và cấu trúc

&emsp;&emsp;Kho lưu trữ này tuân theo quy trình làm việc LLM đầy đủ của ROCm: **cơ sở thống nhất (00-Environment)**, triển khai, tinh chỉnh và các chủ đề kiểu Hạ tầng:

### Bố cục kho lưu trữ

```
hello-rocm/
├── docs/                   # Nguồn tài liệu VitePress
│   ├── zh/                 # Tài liệu tiếng Trung
│   │   ├── 00-environment/ # Cài đặt & cấu hình cơ bản ROCm
│   │   ├── 01-deploy/      # Triển khai LLM trên ROCm
│   │   ├── 02-fine-tune/   # Tinh chỉnh LLM trên ROCm
│   │   ├── 03-infra/       # Tối ưu hóa toán tử & lập trình GPU
│   │   ├── 04-references/  # Tài liệu tham khảo ROCm được tuyển chọn
│   │   └── 05-amd-yes/     # Giới thiệu dự án AMD cộng đồng
│   └── en/                 # Tài liệu tiếng Anh
├── src/                    # Mã nguồn & script
└── assets/                 # Tài nguyên chung
```

### 00. Môi trường — Cơ sở ROCm

<p align="center">
  <strong>🛠️ Cài đặt & cấu hình môi trường ROCm</strong><br>
  <em>Cơ sở duy nhất · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../docs/en/00-environment/index.md">Bắt đầu với Môi trường ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../docs/en/00-environment/rocm-gpu-architecture-table.md">Bản đồ kiến trúc GPU & chỉ mục pip</a><br>
      • Windows 11: trình điều khiển, yêu cầu bảo mật, quy trình cài đặt<br>
      • Ubuntu 24.04: cài đặt dựa trên uv và script cài đặt hợp nhất tùy chọn<br>
      • Xác minh, gỡ cài đặt và chuyển đổi mục tiêu GPU
    </td>
  </tr>
</table>

### 01. Triển khai — Triển khai LLM trên ROCm

<p align="center">
  <strong>🚀 Triển khai LLM trên ROCm</strong><br>
  <em>Từ con số 0 đến một mô hình đang chạy trên GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/01-deploy/index.md">Bắt đầu với Triển khai ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • LM Studio từ đầu<br>
      • vLLM từ đầu<br>
      • Ollama từ đầu<br>
      • llama.cpp từ đầu<br>
      • ATOM từ đầu
    </td>
  </tr>
</table>

### 02. Tinh chỉnh — Tinh chỉnh LLM trên ROCm

<p align="center">
  <strong>🔧 Tinh chỉnh LLM trên ROCm</strong><br>
  <em>Tinh chỉnh hiệu quả trên GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/02-fine-tune/index.md">Bắt đầu với Tinh chỉnh ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Hướng dẫn tinh chỉnh từ đầu<br>
      • Script tinh chỉnh trên một máy<br>
      • Tinh chỉnh đa nút, đa GPU
    </td>
  </tr>
</table>

### 03. Hạ tầng — Tối ưu toán tử & Lập trình GPU

<p align="center">
  <strong>⚙️ Tối ưu toán tử ROCm & Lập trình GPU</strong><br>
  <em>Từ toàn cảnh phần cứng AMD AI đến toán tử HIP và phân tích hiệu suất</em><br>
  📖 <strong><a href="../../docs/en/03-infra/index.md">Bắt đầu với Tối ưu toán tử ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Toàn cảnh phần cứng AMD AI và hệ sinh thái ROCm<br>
      • Phân tích sâu ngăn xếp phần mềm GPU và kiến trúc phần cứng<br>
      • Nhập môn lập trình HIP và thực hành viết kernel thủ công<br>
      • Toán tử PyTorch tùy chỉnh và tích hợp Autograd
    </td>
  </tr>
</table>

### 04. Tài liệu tham khảo

<p align="center">
  <strong>📚 Tài liệu tham khảo ROCm</strong><br>
  <em>Tài nguyên chính thức và cộng đồng</em><br>
  📖 <strong><a href="../../docs/en/04-references/index.md">Tài liệu tham khảo ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">Tài liệu chính thức ROCm</a><br>
      • <a href="https://github.com/amd">AMD trên GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">Ghi chú phát hành ROCm</a><br>
      • <a href="../../docs/en/04-references/index.md#amd-gpu-architecture-whitepapers">Sách trắng kiến trúc GPU AMD (CDNA / RDNA)</a><br>
      • <a href="../../docs/en/04-references/index.md#frameworks-and-inference-services-rocm-quick-install-links">Liên kết cài đặt nhanh ROCm cho framework và dịch vụ suy luận</a><br>
      • Tin tức liên quan
    </td>
  </tr>
</table>

### 05. AMD-YES — Giới thiệu cộng đồng

<p align="center">
  <strong>✨ Giới thiệu dự án AMD</strong><br>
  <em>Các ví dụ do cộng đồng đóng góp trên GPU AMD</em><br>
  📖 <strong><a href="../../docs/en/05-amd-yes/index.md">Bắt đầu với ROCm AMD-YES</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — trình trợ lý dòng lệnh nhẹ<br>
      • Minesweeper Agent — chơi dò mìn với LLM cục bộ (thực hành vòng lặp Agent)<br>
• WeChat "Jump Jump" với YOLOv10 — AI Trò chơi Thực chiến (Huấn luyện và sử dụng yolov10 trên ROCm)<br>
• Chat-甄嬛 — mô hình hội thoại phong cách cổ trang<br>
• Travel planner — demo tác nhân HelloAgents<br>
• Torch-RecHub — hệ thống gợi ý (CTR, thu hồi, đa nhiệm, xuất ONNX)<br>
• happy-llm — huấn luyện LLM phân tán<br>
      • <a href="../../docs/en/05-amd-yes/every-embodied.md">Every Embodied — Sao chép chính sách Embodied AI ROCm</a><br>
    </td>
  </tr>
</table>

## Đóng góp

&emsp;&emsp;Chúng tôi hoan nghênh mọi hình thức đóng góp:

* Cải thiện hoặc thêm hướng dẫn
* Sửa lỗi và bug
* Chia sẻ dự án AMD của bạn
* Đề xuất ý tưởng và hướng phát triển

&emsp;&emsp;Vui lòng đọc **[Content Guide](../../CONTENT_GUIDE_en.md)** (cấu trúc, đặt tên, hình ảnh — tuân thủ các hướng dẫn như Qwen3), sau đó **[CONTRIBUTING_en.md](../../CONTRIBUTING_en.md)** (vấn đề, PR và quy ước thư mục theo từng mô hình).

&emsp;&emsp;Nếu bạn gặp vấn đề về khắc phục sự cố hoặc FAQ khi sử dụng ROCm, triển khai mô hình hoặc đọc hướng dẫn, bạn cũng được chào đón tham gia **[thảo luận cộng đồng](https://zcnijjcepfie.feishu.cn/docx/R2a4dDRUBoo1R2x7mOjcPpPPnOO)** để chia sẻ kinh nghiệm, báo cáo vấn đề và cùng cộng đồng cải thiện hướng dẫn.

&emsp;&emsp;Nếu bạn muốn duy trì kho lưu trữ lâu dài, hãy liên hệ — chúng tôi có thể thêm bạn làm người bảo trì.

## Lời cảm ơn

### Người đóng góp chính

- [Zhixue Song (不要葱姜蒜) — trưởng dự án](https://github.com/KMnO4-zx) (thành viên Datawhale; trưởng dự án self-llm và happy-llm)
- [Yu Chen — trưởng dự án](https://github.com/lucachen) (người sáng tạo nội dung; Google Developer Expert về Học máy)
- [Sizhou Chen — người đóng góp](https://github.com/jjyaoao) (thành viên Datawhale; trưởng dự án hello-agents)
- [Jiahang Pan — người đóng góp](https://github.com/amdjiahangpan) (người sáng tạo nội dung; kỹ sư phần mềm AMD)
- [Weihong Liu — người đóng góp](https://github.com/Weihong-Liu) (thành viên Datawhale)
- [Dongbo Hao — người đóng góp](https://github.com/wlkq151172) (nhà sáng tạo nội dung; nghiên cứu sinh Đại học Jiangnan)
- [Muling Ke — người đóng góp](https://github.com/1985312383) (thành viên Datawhale)

> Luôn hoan nghênh thêm nhiều người đóng góp.

### Khác

- Ý tưởng và phản hồi luôn được chào đón — vui lòng mở issue.
- Cảm ơn tất cả những người đã đóng góp hướng dẫn.
- Cảm ơn **Chương trình Đại học AMD** đã hỗ trợ dự án này.

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/hello-rocm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-rocm" />
  </a>
</div>


## Giấy phép

[Giấy phép MIT](../../LICENSE)

---

<div align="center">

**Cùng nhau xây dựng tương lai AI của AMD.** 💪

Được tạo ra với ❤️ bởi cộng đồng hello-rocm

</div>
