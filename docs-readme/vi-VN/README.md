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


&emsp;&emsp;Kể từ **ROCm 7.10.0** (phát hành ngày 11 tháng 12 năm 2025), ROCm có thể được cài đặt liền mạch trong môi trường ảo Python giống như CUDA, với hỗ trợ chính thức cho cả **Linux và Windows**. Đây là một bước tiến lớn của AMD trong lĩnh vực AI: người học và những người đam mê LLM không còn bị giới hạn bởi phần cứng NVIDIA—GPU AMD là một lựa chọn mạnh mẽ và thiết thực.

&emsp;&emsp;AMD đã cam kết một chu kỳ phát hành ROCm **khoảng sáu tuần** với trọng tâm mạnh mẽ vào AI. Lộ trình phát triển thật sự thú vị.

&emsp;&emsp;Trên toàn thế giới vẫn còn thiếu các hướng dẫn có hệ thống về suy luận, triển khai, huấn luyện, tinh chỉnh và các chủ đề hạ tầng LLM trên ROCm. **hello-rocm** ra đời để lấp đầy khoảng trống đó.

&emsp;&emsp;**Dự án này chủ yếu là các hướng dẫn** để sinh viên và những người thực hành trong tương lai có thể học AMD ROCm một cách có cấu trúc. **Mọi người đều được chào đón mở issue hoặc gửi pull request** để cùng nhau phát triển và duy trì dự án.

> &emsp;&emsp;***Lộ trình học tập: Hoàn thành [00-Môi trường](../../00-Environment/README_EN.md) trước (ROCm + PyTorch + **uv**), sau đó đến triển khai và tinh chỉnh, và cuối cùng là các chủ đề về Infra / cấp độ toán tử. Sau khi môi trường của bạn hoạt động, LM Studio hoặc vLLM là một điểm khởi đầu tốt.***

### Cập nhật mới nhất

- *11/03/2026:* [*Ghi chú phát hành ROCm 7.12.0*](https://rocm.docs.amd.com/en/7.12.0-preview/index.html)

- *11/12/2025:* [*Ghi chú phát hành ROCm 7.10.0*](https://rocm.docs.amd.com/en/7.10.0-preview/about/release-notes.html)

### Các mô hình & hướng dẫn được hỗ trợ

<p align="center">
  <strong>✨ Các LLM chính thống: môi trường · suy luận đa framework · tinh chỉnh ✨</strong><br>
  <em>Thiết lập ROCm thống nhất (Windows / Ubuntu) + ROCm 7+ · hướng dẫn theo từng mô hình (đang phát triển)</em><br>
 <a href="../../00-Environment/README_EN.md">00 — Thiết lập môi trường</a>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">

  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Qwen3</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../01-Deploy/models/Qwen3/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../01-Deploy/models/Qwen3/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../01-Deploy/models/Qwen3/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../01-Deploy/models/Qwen3/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../02-Fine-tune/models/Qwen3/01-Qwen3-0.6B-LoRA及SwanLab可视化记录.md">Qwen3-0.6B LoRA + SwanLab</a><br>
      • <a href="../../02-Fine-tune/models/Qwen3/01-Qwen3-8B-LoRA.ipynb">Qwen3-8B LoRA (Notebook)</a><br>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="border: none !important;"><strong>Gemma4</strong></td>
  </tr>
  <tr>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../01-Deploy/models/Gemma4/gemma4_model.md">Tổng quan mô hình Gemma 4</a><br>
      • <a href="../../01-Deploy/models/Gemma4/lm-studio-rocm7-deploy.md">LM Studio</a><br>
      • <a href="../../01-Deploy/models/Gemma4/vllm-rocm7-deploy.md">vLLM</a><br>
      • <a href="../../01-Deploy/models/Gemma4/ollama-rocm7-deploy.md">Ollama</a><br>
      • <a href="../../01-Deploy/models/Gemma4/llamacpp-rocm7-deploy.md">llama.cpp</a><br>
    </td>
    <td valign="top" width="50%" style="border: none !important;">
      • <a href="../../02-Fine-tune/models/Gemma4/01-Gemma4-E4B-LoRA及SwanLab可视化记录.ipynb">Tinh chỉnh LoRA Gemma4 E4B (TRL, Notebook)</a><br>
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
├── 00-Environment/         # Cài đặt & cấu hình cơ bản ROCm
├── 01-Deploy/              # Triển khai LLM trên ROCm
├── 02-Fine-tune/           # Tinh chỉnh LLM trên ROCm
├── 03-Infra/               # Hạ tầng / toán tử trên ROCm
├── 04-References/          # Tài liệu tham khảo ROCm được tuyển chọn
└── 05-AMD-YES/             # Giới thiệu dự án AMD cộng đồng
```

### 00. Môi trường — Cơ sở ROCm

<p align="center">
  <strong>🛠️ Cài đặt & cấu hình môi trường ROCm</strong><br>
  <em>Cơ sở duy nhất · ROCm 7.12.0 · Windows / Ubuntu · uv + PyTorch</em><br>
  📖 <strong><a href="../../00-Environment/README_VI.md">Bắt đầu với Môi trường ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • <a href="../../00-Environment/rocm-gpu-architecture-table.md">Bản đồ kiến trúc GPU & chỉ mục pip</a><br>
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
  📖 <strong><a href="../../01-Deploy/README_VI.md">Bắt đầu với Triển khai ROCm</a></strong>
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
  📖 <strong><a href="../../02-Fine-tune/README.md">Bắt đầu với Tinh chỉnh ROCm</a></strong>
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

### 03. Hạ tầng — Toán tử & chiều sâu ngăn xếp

<p align="center">
  <strong>⚙️ Hạ tầng & toán tử ROCm</strong><br>
  <em>Từ ngăn xếp phần cứng/phần mềm đến thực hành cấp độ HIP</em><br>
  📖 <strong><a href="../../03-Infra/README.md">Bắt đầu với Hạ tầng ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
      • Di chuyển tự động HIPify<br>
      • Di chuyển thư viện BLAS / DNN (rocBLAS, MIOpen, …)<br>
      • NCCL → RCCL<br>
      • Ánh xạ Nsight → rocprof
    </td>
  </tr>
</table>

### 04. Tài liệu tham khảo

<p align="center">
  <strong>📚 Tài liệu tham khảo ROCm</strong><br>
  <em>Tài nguyên chính thức và cộng đồng</em><br>
  📖 <strong><a href="../../04-References/README.md">Tài liệu tham khảo ROCm</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="100%" align="center" style="border: none !important;">
      • <a href="https://rocm.docs.amd.com/">Tài liệu chính thức ROCm</a><br>
      • <a href="https://github.com/amd">AMD trên GitHub</a><br>
      • <a href="https://rocm.docs.amd.com/en/latest/about/release-notes.html">Ghi chú phát hành ROCm</a><br>
      • Tin tức liên quan
    </td>
  </tr>
</table>

### 05. AMD-YES — Giới thiệu cộng đồng

<p align="center">
  <strong>✨ Giới thiệu dự án AMD</strong><br>
  <em>Các ví dụ do cộng đồng đóng góp trên GPU AMD</em><br>
  📖 <strong><a href="../../05-AMD-YES/README.md">Bắt đầu với ROCm AMD-YES</a></strong>
</p>

<table align="center" border="0" cellspacing="0" cellpadding="0" style="border-collapse: collapse; border: none !important;">
  <tr>
    <td valign="top" width="50%" style="border: none !important;" align="center">
• toy-cli — trình trợ lý dòng lệnh nhẹ<br>
• WeChat “Jump Jump” với YOLOv10 — demo AI trò chơi<br>
• Chat-甄嬛 — mô hình hội thoại phong cách cổ trang<br>
• Travel planner — demo tác nhân HelloAgents<br>
• happy-llm — huấn luyện LLM phân tán
    </td>
  </tr>
</table>

## Đóng góp

&emsp;&emsp;Chúng tôi hoan nghênh mọi hình thức đóng góp:

* Cải thiện hoặc thêm hướng dẫn
* Sửa lỗi và bug
* Chia sẻ dự án AMD của bạn
* Đề xuất ý tưởng và hướng phát triển

&emsp;&emsp;Vui lòng đọc **[规范指南](../../规范指南.md)** (cấu trúc, đặt tên, hình ảnh — tuân thủ các hướng dẫn như Qwen3), sau đó **[CONTRIBUTING.md](../../CONTRIBUTING.md)** (vấn đề, PR và quy ước thư mục theo từng mô hình).

&emsp;&emsp;Nếu bạn muốn duy trì kho lưu trữ lâu dài, hãy liên hệ — chúng tôi có thể thêm bạn làm người bảo trì.

## Lời cảm ơn

### Người đóng góp chính

- [Zhixue Song (不要葱姜蒜) — trưởng dự án](https://github.com/KMnO4-zx) (Datawhale)
- [Yu Chen — trưởng dự án](https://github.com/lucachen) (nội dung — Chuyên gia Google về Học máy)
- [Jiahang Pan — người đóng góp](https://github.com/amdjiahangpan) (nội dung — kỹ sư phần mềm AMD)
- [Weihong Liu — người đóng góp](https://github.com/Weihong-Liu) (Datawhale)

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
