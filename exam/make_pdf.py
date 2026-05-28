import os
import subprocess
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Image, PageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

_here = os.path.dirname(os.path.abspath(__file__))

# регистрируем шрифты DejaVu (поддержка кириллицы)
for _d in ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"]:
    if os.path.exists(os.path.join(_d, "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont("DejaVuMono",      os.path.join(_d, "DejaVuSansMono.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuMono-Bold", os.path.join(_d, "DejaVuSansMono-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans",      os.path.join(_d, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(_d, "DejaVuSans-Bold.ttf")))
        break

# -----------------------------------------------------------------------
# кастомный canvas — нумерация страниц (двухпроходной режим)
# -----------------------------------------------------------------------
class PageNumberCanvas(rl_canvas.Canvas):
    def __init__(self, *args, **kwargs):
        rl_canvas.Canvas.__init__(self, *args, **kwargs)
        self._pages = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self.setFont("DejaVuMono", 8)
            self.setFillColor(colors.grey)
            self.drawRightString(A4[0] - 15 * mm, 8 * mm, f"{self._pageNumber} / {total}")
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

# -----------------------------------------------------------------------
# кастомный DocTemplate — собирает TOC-записи через afterFlowable
# -----------------------------------------------------------------------
class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc_entry"):
            level, text = flowable._toc_entry
            self.notify("TOCEntry", (level, text, self.page))

# параграф, который регистрирует себя в оглавлении
class HeadingPara(Paragraph):
    def __init__(self, text, style, toc_text, level=0):
        Paragraph.__init__(self, text, style)
        self._toc_entry = (level, toc_text)

# -----------------------------------------------------------------------
# стили
# -----------------------------------------------------------------------
title_style = ParagraphStyle("Title", fontName="DejaVuSans-Bold", fontSize=13,
                             leading=18, alignment=TA_CENTER, spaceAfter=4)

section_style = ParagraphStyle("Section", fontName="DejaVuSans-Bold", fontSize=9,
                               leading=13, textColor=colors.Color(0.2, 0.2, 0.5),
                               spaceBefore=8, spaceAfter=4, alignment=TA_LEFT)

code_style = ParagraphStyle("Code", fontName="DejaVuMono", fontSize=7.5, leading=11,
                            backColor=colors.Color(0.96, 0.96, 0.96),
                            leftIndent=4, rightIndent=4, spaceBefore=0, spaceAfter=0,
                            alignment=TA_LEFT, wordWrap=None)

output_style = ParagraphStyle("Output", fontName="DejaVuMono", fontSize=7.5, leading=11,
                              backColor=colors.Color(0.94, 0.97, 0.94),
                              leftIndent=4, rightIndent=4, spaceBefore=0, spaceAfter=0,
                              alignment=TA_LEFT, wordWrap=None)

toc_style = ParagraphStyle("TOC", fontName="DejaVuSans", fontSize=9, leading=14,
                           leftIndent=0, alignment=TA_LEFT)

# стили уровней TOC (reportlab читает их из TableOfContents.levelStyles)
toc_level0 = ParagraphStyle("TOCLevel0", fontName="DejaVuSans", fontSize=9, leading=14,
                            leftIndent=0, rightIndent=0, firstLineIndent=0,
                            spaceBefore=2, spaceAfter=2)

def code_line(text, style):
    text = text.rstrip("\n")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace(" ", "\xa0")
    return Paragraph(text or "\xa0", style)

# -----------------------------------------------------------------------
# читаем исходник, захватываем вывод
# -----------------------------------------------------------------------
src_path = os.path.join(_here, "solution.py")
with open(src_path, encoding="utf-8") as f:
    src_lines = f.readlines()

print("Запускаю solution.py для захвата вывода...")
env = {**os.environ, "MPLBACKEND": "Agg"}
result = subprocess.run(["python", src_path], capture_output=True, text=True, env=env, cwd=_here)
output_lines = result.stdout.splitlines()

# -----------------------------------------------------------------------
# собираем story
# -----------------------------------------------------------------------
pdf_path = os.path.join(_here, "solution.pdf")
doc = ReportDoc(
    pdf_path, pagesize=A4,
    leftMargin=15 * mm, rightMargin=15 * mm,
    topMargin=15 * mm, bottomMargin=18 * mm,
)

story = []

# -- заголовок --
story.append(Spacer(1, 6 * mm))
story.append(Paragraph("М25-604 Аверьянов Никита", title_style))
story.append(Paragraph("Математическая статистика и обработка результатов наблюдений", title_style))
story.append(Paragraph("27.05.2026", title_style))
story.append(Spacer(1, 8 * mm))

# -- оглавление --
toc = TableOfContents()
toc.levelStyles = [toc_level0]
story.append(Paragraph("Содержание", section_style))
story.append(Spacer(1, 2 * mm))
story.append(toc)
story.append(PageBreak())

# -- код --
story.append(HeadingPara("Исходный код", section_style, "Исходный код"))
for line in src_lines:
    story.append(code_line(line, code_style))

# -- вывод --
story.append(PageBreak())
story.append(HeadingPara("Вывод программы", section_style, "Вывод программы"))
story.append(Spacer(1, 2 * mm))
for line in output_lines:
    story.append(code_line(line, output_style))

# -- графики --
graph_files = sorted([f for f in os.listdir(_here) if f.startswith("graph_") and f.endswith(".png")])
page_w = A4[0] - 30 * mm
page_h = A4[1] - 30 * mm
for gf in graph_files:
    num = gf.replace("graph_", "").replace(".png", "")
    story.append(PageBreak())
    label = f"График {num}"
    story.append(HeadingPara(label, section_style, label))
    story.append(Spacer(1, 2 * mm))
    img = Image(os.path.join(_here, gf), width=page_w, height=page_h, kind="proportional")
    story.append(img)

# двухпроходной билд — первый проход собирает номера страниц, второй рендерит TOC
doc.multiBuild(story, canvasmaker=PageNumberCanvas)
print(f"PDF сохранён: {pdf_path}")
