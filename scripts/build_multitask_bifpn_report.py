from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\35297\Desktop")
OUT_DIR = ROOT / "output" / "pdf"
OUT_PDF = OUT_DIR / "multitask_bifpn_report.pdf"
FONT_REGULAR = "DengXian"
FONT_BOLD = "DengXian-Bold"
FONT_REGULAR_PATH = r"C:\Windows\Fonts\Deng.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\Dengb.ttf"


def load_json(name: str) -> dict:
    with (DESKTOP / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "-"
    return f"{100.0 * float(v):.{digits}f}"


def num(v: float | int | None) -> str:
    if v is None:
        return "-"
    return f"{int(v):,}"


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 0.75 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_table(data, col_widths=None, font_size=8.2, leading=10.5):
    wrapped = []
    body_style = ParagraphStyle(
        "table_body",
        fontName=FONT_REGULAR,
        fontSize=font_size,
        leading=leading,
        wordWrap="CJK",
    )
    head_style = ParagraphStyle(
        "table_head",
        fontName=FONT_REGULAR,
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    for r, row in enumerate(data):
        wrapped.append([p(str(cell), head_style if r == 0 else body_style) for cell in row])
    table = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8CDD4")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def main() -> None:
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, FONT_REGULAR_PATH))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, FONT_BOLD_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yolov8_det = load_json("yolov8-det.json")
    yolov8_pose = load_json("yolov8-pose.json")
    bifpn_det = load_json("bifpn_detect_best.json")
    bifpn_dual = load_json("bifpn_dual_final_cocoeval_metrics.json")
    bifpn_dual_det_best = load_json("bifpn_dual_det_best.json")

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName=FONT_BOLD,
        fontSize=20,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"),
        wordWrap="CJK",
    )
    subtitle = ParagraphStyle(
        "subtitle",
        parent=styles["Normal"],
        fontName=FONT_REGULAR,
        fontSize=10,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5563"),
        wordWrap="CJK",
    )
    h1 = ParagraphStyle(
        "h1",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=14,
        leading=20,
        spaceBefore=12,
        spaceAfter=8,
        textColor=colors.HexColor("#111827"),
        wordWrap="CJK",
    )
    h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=11.5,
        leading=16,
        spaceBefore=9,
        spaceAfter=5,
        textColor=colors.HexColor("#1F2937"),
        wordWrap="CJK",
    )
    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontName=FONT_REGULAR,
        fontSize=10.2,
        leading=17,
        firstLineIndent=20,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        wordWrap="CJK",
    )
    small = ParagraphStyle(
        "small",
        parent=styles["BodyText"],
        fontName=FONT_REGULAR,
        fontSize=8.8,
        leading=13,
        textColor=colors.HexColor("#4B5563"),
        wordWrap="CJK",
    )

    det_single_ap = bifpn_det["bbox/AP"]
    pose_single_ap = load_json("yolov8-pose.json")["keypoints/AP"]
    bifpn_pose_ap = 0.6503568561462488
    dual_det_ratio = bifpn_dual["bbox/AP"] / det_single_ap
    dual_pose_ratio = bifpn_dual["keypoints/AP"] / bifpn_pose_ap
    det_best_ratio = bifpn_dual_det_best["bbox/AP"] / det_single_ap
    det_best_pose_ratio = bifpn_dual_det_best["keypoints/AP"] / bifpn_pose_ap

    story = []
    story.append(p("基于 BiFPN 的检测-姿态多任务模型实验报告", title))
    story.append(Spacer(1, 0.25 * cm))
    story.append(p("实验目的、融合训练策略与目标领域微调方案", subtitle))
    story.append(Spacer(1, 0.5 * cm))
    story.append(p("摘要", h1))
    story.append(
        p(
            "本文围绕同等参数规模下的目标检测与人体姿态估计融合问题展开。首先以 YOLOv8m 作为参照，"
            "在保持模型规模接近的前提下引入 BiFPN 作为多尺度特征融合结构，并分别完成检测单任务与姿态单任务训练。"
            "单任务结果表明，BiFPN 检测模型在 COCO bbox AP 上达到 50.57%，略高于 YOLOv8m 的 49.54%；"
            "BiFPN 姿态模型在 keypoints AP 上达到 65.04%，与 YOLOv8m-pose 的 64.99% 基本一致。"
            "在此基础上进行检测-姿态融合训练后，姿态精度保持充分，但检测精度下降明显。后续实验因此集中于提升融合模型中的检测能力，"
            "并进一步设计了面向火焰、水、抽烟、安全帽、跌倒与挥手等目标领域任务的微调方案。",
            body,
        )
    )

    story.append(p("一、研究动机与实验逻辑", h1))
    paragraphs = [
        "多任务模型的目标是在一次前向推理中同时获得目标检测与人体关键点结果，从而降低部署成本并简化后处理流程。"
        "然而，检测任务与姿态任务虽然共享图像特征，但优化目标并不完全一致：检测更依赖类别判别与全类别框回归，"
        "姿态则更依赖人体局部结构与关键点几何约束。因此，多任务融合并不是简单叠加两个输出头，而需要验证共享特征是否会产生任务竞争。",
        "为避免模型容量差异造成不公平比较，实验首先以 YOLOv8m 作为同参数量参考，并选择 BiFPN 作为特征融合结构。"
        "BiFPN 的作用是加强不同尺度特征之间的双向信息传递，使高层语义与低层定位信息能够反复融合。"
        "在检测和姿态两个单任务上分别训练 BiFPN 后，可以判断该结构本身是否具备足够的单任务表达能力。",
        "单任务验证通过后，再进行双头融合训练。融合模型共享 backbone 与 neck，并分别输出检测结果和姿态结果。"
        "实验发现，融合模型的姿态任务可以达到甚至略高于单任务姿态水平，而检测任务低于单检测模型。"
        "这说明当前主要瓶颈并不是姿态分支退化，而是检测分支在共享特征下受到任务竞争或优化偏置影响。",
        "因此，后续实验围绕三个方向展开：其一，通过任务权重或动态权重改变不同任务对共享参数的优化强度；"
        "其二，通过 PCGrad 等梯度冲突处理方法缓解两个任务在共享参数上的更新冲突；"
        "其三，通过检测单任务教师模型进行蒸馏，使融合模型在训练过程中保留单检测模型已经学习到的判别能力。",
    ]
    for text in paragraphs:
        story.append(p(text, body))

    story.append(p("二、评价指标与实验结果", h1))
    story.append(p("2.1 指标含义", h2))
    metric_table = [
        ["正式名称", "记号", "含义"],
        ["平均精度", "AP@[0.50:0.95]", "在 IoU 或 OKS 阈值 0.50 到 0.95 间取平均，反映整体定位与分类质量。"],
        ["宽松阈值平均精度", "AP@0.50", "匹配阈值为 0.50 时的 AP，主要反映召回与粗定位能力。"],
        ["严格阈值平均精度", "AP@0.75", "匹配阈值为 0.75 时的 AP，更强调定位精度。"],
        ["尺度分组平均精度", "AP small/medium/large", "分别统计小、中、大目标上的 AP，用于分析尺度敏感性。"],
        ["平均召回率", "AR", "在给定最大检测数限制下的平均召回能力。检测任务使用 AR@1/10/100，姿态任务使用 keypoints AR。"],
        ["关键点平均精度", "keypoints AP", "基于 OKS 的关键点 AP，考虑关键点位置误差、目标尺度和可见性。"],
        ["相对单任务比例", "Ratio to single-task", "融合模型相对于对应 BiFPN 单任务模型的 AP 比例，用于衡量多任务融合后的性能保留程度。"],
    ]
    story.append(make_table(metric_table, col_widths=[3.2 * cm, 3.0 * cm, 9.4 * cm], font_size=8.3))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        p(
            "检测任务采用 COCO2017 val 全量 5000 张图像评价；姿态单任务采用 person keypoints 评价集合。"
            "双头模型在 COCO2017 val 上同时输出 bbox 与 keypoints，因此可同时观察检测与姿态的变化。",
            body,
        )
    )

    story.append(p("2.2 检测任务结果", h2))
    det_rows = [
        [
            "实验",
            "权重选择",
            "验证图像",
            "预测框数",
            "bbox AP",
            "AP50",
            "AP75",
            "AP small",
            "AP medium",
            "AP large",
            "AR@100",
            "相对 BiFPN 单检",
        ],
        [
            "YOLOv8m 检测",
            "官方权重",
            num(yolov8_det["num_images"]),
            num(yolov8_det["num_bbox_predictions"]),
            pct(yolov8_det["bbox/AP"]),
            pct(yolov8_det["bbox/AP50"]),
            pct(yolov8_det["bbox/AP75"]),
            pct(yolov8_det["bbox/AP_small"]),
            pct(yolov8_det["bbox/AP_medium"]),
            pct(yolov8_det["bbox/AP_large"]),
            pct(yolov8_det["bbox/AR_100"]),
            pct(yolov8_det["bbox/AP"] / det_single_ap),
        ],
        [
            "BiFPN 单检测",
            "loss best",
            num(bifpn_det["num_images"]),
            num(bifpn_det["num_bbox_predictions"]),
            pct(bifpn_det["bbox/AP"]),
            pct(bifpn_det["bbox/AP50"]),
            pct(bifpn_det["bbox/AP75"]),
            pct(bifpn_det["bbox/AP_small"]),
            pct(bifpn_det["bbox/AP_medium"]),
            pct(bifpn_det["bbox/AP_large"]),
            pct(bifpn_det["bbox/AR_100"]),
            "100.00",
        ],
        [
            "BiFPN 双头融合",
            "loss best",
            num(bifpn_dual["num_images"]),
            num(bifpn_dual["num_bbox_predictions"]),
            pct(bifpn_dual["bbox/AP"]),
            pct(bifpn_dual["bbox/AP50"]),
            pct(bifpn_dual["bbox/AP75"]),
            pct(bifpn_dual["bbox/AP_small"]),
            pct(bifpn_dual["bbox/AP_medium"]),
            pct(bifpn_dual["bbox/AP_large"]),
            pct(bifpn_dual["bbox/AR_100"]),
            pct(dual_det_ratio),
        ],
        [
            "BiFPN 双头融合",
            "det metric best",
            num(bifpn_dual_det_best["num_images"]),
            num(bifpn_dual_det_best["num_bbox_predictions"]),
            pct(bifpn_dual_det_best["bbox/AP"]),
            pct(bifpn_dual_det_best["bbox/AP50"]),
            pct(bifpn_dual_det_best["bbox/AP75"]),
            pct(bifpn_dual_det_best["bbox/AP_small"]),
            pct(bifpn_dual_det_best["bbox/AP_medium"]),
            pct(bifpn_dual_det_best["bbox/AP_large"]),
            pct(bifpn_dual_det_best["bbox/AR_100"]),
            pct(det_best_ratio),
        ],
    ]
    story.append(make_table(det_rows, col_widths=[2.5 * cm, 2.0 * cm, 1.45 * cm, 1.65 * cm, 1.4 * cm, 1.25 * cm, 1.25 * cm, 1.25 * cm, 1.35 * cm, 1.25 * cm, 1.25 * cm, 1.45 * cm], font_size=6.7, leading=8.4))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        p(
            "从检测结果看，BiFPN 单检测 AP 为 50.57%，相较 YOLOv8m 的 49.54% 略有提升，说明该结构在检测单任务上是有效的。"
            "融合训练后，loss best 的检测 AP 为 45.14%，约为 BiFPN 单检测的 89.27%；按检测指标保存的 det best 提升至 46.84%，"
            "约为单检测的 92.62%。因此，改进保存策略和面向检测的优化确实带来增益，但仍未完全恢复单检测性能。",
            body,
        )
    )

    story.append(PageBreak())
    story.append(p("2.3 姿态任务结果", h2))
    pose_rows = [
        [
            "实验",
            "权重选择",
            "验证图像",
            "关键点预测数",
            "keypoints AP",
            "AP50",
            "AP75",
            "AP medium",
            "AP large",
            "AR",
            "AR50",
            "相对 BiFPN 单姿态",
        ],
        [
            "YOLOv8m-pose",
            "官方权重",
            num(yolov8_pose["num_images"]),
            num(yolov8_pose["num_keypoint_predictions"]),
            pct(yolov8_pose["keypoints/AP"]),
            pct(yolov8_pose["keypoints/AP50"]),
            pct(yolov8_pose["keypoints/AP75"]),
            pct(yolov8_pose["keypoints/AP_medium"]),
            pct(yolov8_pose["keypoints/AP_large"]),
            pct(yolov8_pose["keypoints/AR"]),
            pct(yolov8_pose["keypoints/AR50"]),
            pct(yolov8_pose["keypoints/AP"] / bifpn_pose_ap),
        ],
        [
            "BiFPN 单姿态",
            "loss best",
            "2,346",
            "33,571",
            pct(bifpn_pose_ap),
            "88.20",
            "72.28",
            "60.06",
            "73.55",
            "71.72",
            "91.88",
            "100.00",
        ],
        [
            "BiFPN 双头融合",
            "loss best",
            num(bifpn_dual["num_images"]),
            num(bifpn_dual["num_keypoint_predictions"]),
            pct(bifpn_dual["keypoints/AP"]),
            pct(bifpn_dual["keypoints/AP50"]),
            pct(bifpn_dual["keypoints/AP75"]),
            pct(bifpn_dual["keypoints/AP_medium"]),
            pct(bifpn_dual["keypoints/AP_large"]),
            pct(bifpn_dual["keypoints/AR"]),
            pct(bifpn_dual["keypoints/AR50"]),
            pct(dual_pose_ratio),
        ],
        [
            "BiFPN 双头融合",
            "det metric best",
            num(bifpn_dual_det_best["num_images"]),
            num(bifpn_dual_det_best["num_keypoint_predictions"]),
            pct(bifpn_dual_det_best["keypoints/AP"]),
            pct(bifpn_dual_det_best["keypoints/AP50"]),
            pct(bifpn_dual_det_best["keypoints/AP75"]),
            pct(bifpn_dual_det_best["keypoints/AP_medium"]),
            pct(bifpn_dual_det_best["keypoints/AP_large"]),
            pct(bifpn_dual_det_best["keypoints/AR"]),
            pct(bifpn_dual_det_best["keypoints/AR50"]),
            pct(det_best_pose_ratio),
        ],
    ]
    story.append(make_table(pose_rows, col_widths=[2.5 * cm, 2.0 * cm, 1.45 * cm, 1.65 * cm, 1.45 * cm, 1.25 * cm, 1.25 * cm, 1.35 * cm, 1.25 * cm, 1.15 * cm, 1.15 * cm, 1.55 * cm], font_size=6.7, leading=8.4))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        p(
            "姿态结果显示，BiFPN 单姿态与 YOLOv8m-pose 基本持平。双头融合的 loss best keypoints AP 为 65.24%，"
            "相当于单姿态的 100.31%，说明融合训练并未削弱姿态能力。det best 的姿态 AP 为 63.63%，"
            "相当于单姿态的 97.85%，表明偏向检测的保存和训练策略会带来一定姿态代价，但该代价仍处于可控范围。",
            body,
        )
    )

    story.append(p("2.4 阶段性结论", h2))
    story.append(
        p(
            "当前结果支持三个判断。第一，BiFPN 并非导致检测下降的原因，因为单检测已经超过 YOLOv8m。"
            "第二，融合模型中的姿态任务具有较强稳定性，甚至可以达到单任务水平。第三，融合模型的主要矛盾是检测任务在共享特征和联合优化下的性能损失。"
            "因此，后续优化应优先围绕检测分支的监督强度、检测教师知识约束、共享梯度冲突处理以及模型保存准则展开。",
            body,
        )
    )

    story.append(p("三、融合训练中的优化策略", h1))
    methods = [
        (
            "固定任务权重",
            "最基础的融合训练将检测损失与姿态损失按固定比例相加。该策略实现简单，便于控制训练偏向。"
            "在当前问题中，提高检测权重可以增加检测任务对共享特征的更新强度，但过高的检测权重可能牺牲姿态稳定性，"
            "因此需要结合验证指标而不是只看训练损失选择权重。",
        ),
        (
            "动态权重 DWA",
            "Dynamic Weight Averaging 根据相邻训练阶段中各任务损失下降速度调整权重。若某个任务下降较慢，说明其相对训练进展不足，"
            "该任务会获得更高权重；若下降较快，则权重降低。该方法适合在任务尺度基本稳定时使用。"
            "在当前融合训练中，DWA 用于在检测与姿态之间自动分配损失权重，但由于两个任务后期损失变化趋于平缓，权重波动有限，"
            "因此其实际调节能力较弱。",
        ),
        (
            "PCGrad",
            "PCGrad 从梯度方向角度处理多任务冲突。当检测损失和姿态损失在共享参数上的梯度方向相互冲突时，"
            "PCGrad 会去除其中与另一任务相反的梯度分量，从而减少一个任务更新对另一个任务的直接破坏。"
            "该方法更关注共享 backbone 与 neck 的更新方向，适合用于双任务共享特征学习阶段。",
        ),
        (
            "GradNorm 与 CAGrad",
            "GradNorm 通过约束不同任务的梯度范数，使训练速度较慢的任务获得更强更新；CAGrad 则在多任务梯度之间寻找更保守的组合方向，"
            "目标是在提升整体目标的同时降低任务间冲突。这两类方法可作为 PCGrad 的补充实验，用于判断检测下降是否主要来自梯度冲突。",
        ),
        (
            "检测蒸馏",
            "检测蒸馏使用已经训练好的单检测模型作为教师模型，并在融合模型训练时约束学生检测分支的类别分布和框分布。"
            "其核心思想是：单检测模型已经具备较强检测判别能力，融合模型不应只依赖原始标签重新学习检测，而应同时学习教师模型的输出结构。"
            "在当前实验中，蒸馏主要作用于检测分支，目标是在保持姿态精度的同时缓解检测 AP 的下降。",
        ),
        (
            "基于验证指标的保存策略",
            "仅按验证损失保存权重并不一定对应 COCO AP 最优，尤其在多任务中，不同损失项的尺度和下降趋势不同。"
            "因此需要同时保存 loss best、检测指标 best、姿态指标 best 以及两任务综合指标 best。"
            "本次 det best 的结果已经说明，检测指标保存能够得到比 loss best 更高的检测 AP。",
        ),
    ]
    method_rows = [["策略", "原理与当前应用"]]
    for name, desc in methods:
        method_rows.append([name, desc])
    story.append(make_table(method_rows, col_widths=[3.2 * cm, 13.0 * cm], font_size=8.2, leading=12))

    story.append(PageBreak())
    story.append(p("四、目标领域微调方案", h1))
    story.append(
        p(
            "在已有双头模型基础上，目标领域任务被重新划分为三类：第一类是人体与关键点，继续由原姿态分支承担；"
            "第二类是火焰与水等非人体目标，由新的领域检测分支承担；第三类是依附于人的行为或状态属性，"
            "包括抽烟、跌倒、挥手和是否佩戴安全帽，由新增的人体属性分支承担。该设计的核心原则是尽量保留原模型已经学好的姿态能力，"
            "同时为领域任务增加必要的新输出空间。",
            body,
        )
    )
    story.append(p("4.1 模型改进", h2))
    for text in [
        "原姿态分支继续用于输出人体框和关键点，并加载已有双头模型中的对应权重。由于该分支在 COCO 姿态任务上已经达到较高精度，微调初期不应大幅扰动。",
        "原通用检测任务与目标领域的火焰、水检测类别并不一致，因此领域检测分支采用新的检测头进行学习。这样可以避免将 COCO 通用类别空间强行映射到领域类别空间。",
        "人体属性分支以检测到的人为基本单位，输出多标签属性。抽烟、跌倒、挥手和安全帽并不是互斥类别，同一个人可以同时具有多个属性，因此属性学习采用多标签二分类形式。",
        "领域检测头和属性头属于新增模块，初始阶段需要较大学习率适配；姿态分支属于已学习模块，后续只在共享 neck 被解冻或全模型微调时作为约束参与训练。",
    ]:
        story.append(p(text, body))

    story.append(p("4.2 数据处理与监督构造", h2))
    data_rows = [
        ["数据来源", "主要监督信号", "处理原则"],
        ["跌倒数据", "falling、waving 属性", "清洗无关标签，将跌倒和睡卧类样本归入 falling；属性与人框一对一匹配，避免同一属性重复分配给多人。"],
        ["安全帽数据", "helmet_on 属性", "保留与安全帽状态相关的标注，利用人框与头盔区域关系构造人的属性标签；难以确定的样本可置为 unknown。"],
        ["抽烟数据", "smoking 属性", "该数据集中的人默认作为抽烟正样本；其他非抽烟数据集中的人可作为 smoking 负样本以增加类别平衡。"],
        ["火焰与水数据", "fire、water 检测", "剔除无关类别，只保留领域检测目标；若清洗后图像无有效标注则不进入训练。"],
        ["COCO person pose", "人体框与关键点", "作为姿态保持数据，防止领域微调过程中姿态分支退化。"],
    ]
    story.append(make_table(data_rows, col_widths=[3.0 * cm, 3.4 * cm, 9.8 * cm], font_size=8.2, leading=11.5))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        p(
            "属性分配以重新生成的人体框和关键点为基准。每个属性标注只分配给一个最匹配的人，通常依据重叠面积或匹配分数选择最合适的人体目标，"
            "并在分配后将该属性实例标记为已使用。这样可以避免多人靠近时，一个属性标签被重复赋给多个对象。对于缺少可靠证据的属性，不强行作为负样本，"
            "而是保留为 unknown；对于语义明确的非目标数据集，则可以引入负样本以缓解正负极不平衡。",
            body,
        )
    )

    story.append(p("4.3 四阶段微调策略", h2))
    stage_rows = [
        ["阶段", "训练对象", "训练目的"],
        ["第一阶段：新增头适配", "领域检测头、属性头", "在不改变已有姿态能力的前提下，使新增输出头先适应目标领域标签空间。"],
        ["第二阶段：检测适配", "检测适配层、领域检测头、属性头", "进一步增强火焰和水检测能力，同时继续训练属性分支。"],
        ["第三阶段：neck 受控解冻", "neck、检测适配层、领域检测头、属性头、姿态分支", "当共享特征开始变化时引入姿态损失，防止人体框和关键点性能明显退化。"],
        ["第四阶段：全模型微调", "全模型小学习率微调", "在已有适配基础上统一调整 backbone、neck 与各任务头，使领域任务和姿态保持达到更稳定的平衡。"],
    ]
    story.append(make_table(stage_rows, col_widths=[3.2 * cm, 4.2 * cm, 8.8 * cm], font_size=8.2, leading=11.5))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        p(
            "该微调方案采用由浅到深的解冻顺序。前两个阶段只训练新增或轻量适配模块，是为了避免随机初始化的新任务头破坏已有特征。"
            "第三阶段开始解冻 neck，此时必须加入姿态损失，因为 neck 的变化会影响姿态分支输入。第四阶段再进行全模型小学习率微调，"
            "用于消除分阶段训练造成的模块间不一致。训练完成后，需要重新进行 COCO keypoints 评价，以确认姿态能力相对原双头模型没有不可接受的下降；"
            "领域检测与属性任务则应在目标领域验证集上分别评估。",
            body,
        )
    )

    story.append(p("4.4 微调后验证结果", h2))
    story.append(
        p(
            "微调后模型在目标领域验证集上进行领域检测和属性评价，同时在 COCO person keypoints 评价集上重新测试姿态能力。"
            "领域验证集共 909 张图像；COCO 姿态评价集共 2346 张图像。由于当前目标领域验证集中 water 类没有标注实例，"
            "领域检测指标主要反映 fire 类检测表现。",
            body,
        )
    )
    domain_rows = [
        ["任务", "指标", "结果", "说明"],
        ["领域检测", "mAP@0.50", "35.44%", "当前验证集主要包含 fire 类，water 类 GT 为 0。"],
        ["领域检测", "mAP@[0.50:0.95]", "12.50%", "严格定位阈值下仍有提升空间，说明框定位质量和置信度排序仍需继续优化。"],
        ["领域检测", "Precision / Recall / F1", "48.24% / 35.86% / 41.14%", "在 domain_conf=0.25 下统计，召回偏低是当前主要问题。"],
        ["fire", "AP@0.50 / AP@[0.50:0.95]", "35.44% / 12.50%", "fire GT 为 343，置信度阈值下 TP=123、FP=132、FN=220。"],
        ["water", "AP", "-", "验证集中没有 water 标注，因此不能据此判断 water 检测能力。"],
    ]
    story.append(make_table(domain_rows, col_widths=[2.4 * cm, 4.0 * cm, 3.6 * cm, 6.2 * cm], font_size=8.1, leading=11.5))
    story.append(Spacer(1, 0.2 * cm))

    attr_rows = [
        ["属性", "正样本数", "Precision", "Recall", "F1", "Accuracy", "主要观察"],
        ["smoking", "39", "100.00%", "97.44%", "98.70%", "99.96%", "抽烟属性表现稳定，误报为 0。"],
        ["falling", "163", "92.86%", "95.71%", "94.26%", "99.21%", "跌倒属性已具备较高可用性。"],
        ["waving", "7", "42.86%", "42.86%", "42.86%", "99.67%", "正样本极少，accuracy 不具备代表性，应重点增加挥手样本。"],
        ["helmet_on", "258", "96.56%", "98.06%", "97.31%", "98.92%", "安全帽属性表现较好，但仍有 1107 个 unknown 样本未参与该属性统计。"],
        ["micro", "467", "94.74%", "96.36%", "95.54%", "99.51%", "整体属性性能较高，但受类别不平衡影响，需要结合单属性 F1 判断。"],
    ]
    story.append(make_table(attr_rows, col_widths=[2.3 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm, 1.7 * cm, 1.8 * cm, 4.8 * cm], font_size=7.6, leading=10.5))
    story.append(Spacer(1, 0.2 * cm))

    pose_keep_rows = [
        ["模型", "评价集", "keypoints AP", "AP50", "AP75", "AR", "相对微调前双头"],
        ["微调前双头 loss best", "COCO person keypoints", "65.24%", "88.48%", "71.99%", "72.16%", "100.00%"],
        ["领域微调后模型", "COCO person keypoints", "64.71%", "88.29%", "71.34%", "71.17%", "99.19%"],
    ]
    story.append(make_table(pose_keep_rows, col_widths=[3.4 * cm, 4.0 * cm, 2.0 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.9 * cm], font_size=7.8, leading=10.5))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        p(
            "从结果看，领域微调后的属性任务整体较好，尤其 smoking、falling 和 helmet_on 的 F1 均接近或超过 94%。"
            "waving 的 F1 较低，主要原因是验证集中正样本仅 7 个，数据规模不足会导致指标波动明显。"
            "领域检测中 fire 的 mAP@0.50 为 35.44%，但召回率仅 35.86%，说明当前模型漏检仍较多，后续应优先补充火焰和水的验证样本，"
            "并继续优化领域检测分支。姿态保持方面，微调后 COCO keypoints AP 为 64.71%，相对微调前双头模型约保持 99.19%，"
            "说明四阶段微调策略基本达到了在新增领域能力的同时保持姿态能力的目的。",
            body,
        )
    )

    story.append(p("五、综合结论", h1))
    for text in [
        "BiFPN 单任务实验表明，在接近 YOLOv8m 的模型规模下，检测与姿态两个单任务均可以达到 YOLOv8m 系列的水平，因此 BiFPN 是合理的融合基础。",
        "双头融合后的主要问题集中在检测任务。loss best 模型的检测 AP 为 45.14%，det best 模型提升至 46.84%，但仍低于 BiFPN 单检测的 50.57%。",
        "姿态任务在融合中表现稳定，loss best 下 keypoints AP 为 65.24%，略高于 BiFPN 单姿态；det best 下下降到 63.63%，但仍保持了较高水平。",
        "领域微调后，模型在目标领域属性任务上取得较好结果，属性 micro F1 为 95.54%；COCO keypoints AP 为 64.71%，相对微调前双头模型保持约 99.19%。当前领域检测仍以 fire 为主，mAP@0.50 为 35.44%，后续需要补充 water 验证样本并继续提高召回率。",
        "后续优化应继续以检测恢复为核心，同时设置姿态保持约束。目标领域微调时，应将火焰和水作为新的领域检测任务，将抽烟、跌倒、挥手和安全帽作为人属性任务，并通过分阶段解冻降低对原姿态能力的破坏。",
    ]:
        story.append(p(text, body))

    story.append(p("参考方法", h1))
    refs = [
        "Tan 等提出的 EfficientDet/BiFPN 思路：通过可学习的双向特征金字塔增强多尺度融合。",
        "Liu 等提出的 Dynamic Weight Averaging：依据各任务损失下降速度动态调整多任务权重。",
        "Yu 等提出的 PCGrad：在多任务梯度冲突时执行梯度投影以减少负迁移。",
        "Chen 等提出的 GradNorm：通过梯度范数平衡不同任务的训练速度。",
        "Liu 等提出的 CAGrad：通过冲突规避的梯度组合提升多任务优化稳定性。",
        "Hinton 等提出的知识蒸馏：使用教师模型输出作为学生模型的软监督信号。",
    ]
    for item in refs:
        story.append(p(item, small))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        p(
            "数据文件：yolov8-det.json、yolov8-pose.json、bifpn_detect_best.json、"
            "bifpn_dual_final_cocoeval_metrics.json、bifpn_dual_det_best.json，以及领域微调验证结果。",
            small,
        )
    )

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.45 * cm,
        title="基于 BiFPN 的检测-姿态多任务模型实验报告",
        author="AI4PumpRoom",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
