"""PlangoGraph 12-Slide External Evaluation Deck Builder

Generates docs/plangograph_evaluation_12slides.pptx directly using OpenXML
PresentationML specifications. Delivers 16:9 widescreen slides with navy bookends,
off-white content styling, tables, cards, flowcharts, metrics and embedded images.
Enforces British/Commonwealth spelling, zero em/en-dashes in prose and no AI crutches.
"""

from __future__ import annotations

import html
import io
import os
import shutil
import zipfile
from typing import Any, Dict, List, Optional, Tuple


def escape_xml(s: str) -> str:
    """Escapes XML entities in text strings."""
    return html.escape(str(s), quote=True)


class PresentationBuilder:
    """Constructs ECMA-376 OpenXML PPTX packages directly."""

    # 16:9 Widescreen dimensions in English Metric Units (EMUs)
    WIDTH_EMU = 12192000   # 13.333333 inches * 914400
    HEIGHT_EMU = 6858000   # 7.5 inches * 914400

    def __init__(self) -> None:
        self.slides_xml: List[str] = []
        self.slide_rels: List[List[Dict[str, str]]] = []
        self.media_files: Dict[str, bytes] = {}

    def add_slide(self, bg_color: str = "F8FAFC") -> int:
        """Initialises a new slide with a background colour."""
        slide_idx = len(self.slides_xml)
        self.slides_xml.append("")  # placeholder, set when finalised
        self.slide_rels.append([
            {
                "id": "rId1",
                "type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
                "target": "../slideLayouts/slideLayout1.xml",
            }
        ])
        return slide_idx

    def add_image_rel(self, slide_idx: int, media_filename: str) -> str:
        """Adds an image relationship to a slide and returns the relationship id."""
        rel_id = f"rId{len(self.slide_rels[slide_idx]) + 1}"
        self.slide_rels[slide_idx].append({
            "id": rel_id,
            "type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "target": f"../media/{media_filename}",
        })
        return rel_id

    def embed_media(self, media_filename: str, file_path: str) -> None:
        """Reads a local binary file into the package media pool."""
        with open(file_path, "rb") as f:
            self.media_files[media_filename] = f.read()

    def set_slide_content(self, slide_idx: int, bg_color: str, shapes_xml: str) -> None:
        """Wraps shapes into a complete slide XML document."""
        xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="{bg_color}"/></a:solidFill>
        <a:effectLst/>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {shapes_xml}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""
        self.slides_xml[slide_idx] = xml

    def save(self, output_path: str) -> None:
        """Compiles the complete presentation package into a ZIP archive."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        buf = io.BytesIO()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            # 1. [Content_Types].xml
            ct_overrides = []
            for i in range(1, len(self.slides_xml) + 1):
                ct_overrides.append(
                    f'  <Override PartName="/ppt/slides/slide{i}.xml" '
                    'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                )

            content_types = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
{chr(10).join(ct_overrides)}
</Types>"""
            z.writestr("[Content_Types].xml", content_types)

            # 2. _rels/.rels
            root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""
            z.writestr("_rels/.rels", root_rels)

            # 3. ppt/presentation.xml
            sld_ids = []
            for i in range(1, len(self.slides_xml) + 1):
                sld_ids.append(f'    <p:sldId id="{255 + i}" r:id="rId{i + 1}"/>')

            presentation_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
{chr(10).join(sld_ids)}
  </p:sldIdLst>
  <p:sldSz cx="{self.WIDTH_EMU}" cy="{self.HEIGHT_EMU}" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>"""
            z.writestr("ppt/presentation.xml", presentation_xml)

            # 4. ppt/_rels/presentation.xml.rels
            pres_rels = [
                '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
            ]
            for i in range(1, len(self.slides_xml) + 1):
                pres_rels.append(
                    f'  <Relationship Id="rId{i + 1}" '
                    f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
                    f'Target="slides/slide{i}.xml"/>'
                )

            pres_rels_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{chr(10).join(pres_rels)}
</Relationships>"""
            z.writestr("ppt/_rels/presentation.xml.rels", pres_rels_xml)

            # 5. ppt/slideMasters/slideMaster1.xml
            slide_master_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2"
            accent1="accent1" accent2="accent2" accent3="accent3"
            accent4="accent4" accent5="accent5" accent6="accent6"
            hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
</p:sldMaster>"""
            z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master_xml)

            # 6. ppt/slideMasters/_rels/slideMaster1.xml.rels
            master_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""
            z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels_xml)

            # 7. ppt/slideLayouts/slideLayout1.xml
            layout_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""
            z.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml)

            # 8. ppt/slideLayouts/_rels/slideLayout1.xml.rels
            layout_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""
            z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels_xml)

            # 9. ppt/theme/theme1.xml
            theme_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="PlangoGraph Modern">
  <a:themeElements>
    <a:clrScheme name="PlangoScheme">
      <a:dk1><a:srgbClr val="0F172A"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1E293B"/></a:dk2>
      <a:lt2><a:srgbClr val="F8FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1>
      <a:accent2><a:srgbClr val="38BDF8"/></a:accent2>
      <a:accent3><a:srgbClr val="059669"/></a:accent3>
      <a:accent4><a:srgbClr val="DC2626"/></a:accent4>
      <a:accent5><a:srgbClr val="D97706"/></a:accent5>
      <a:accent6><a:srgbClr val="4F46E5"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="1D4ED8"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="ModernFonts">
      <a:majorFont>
        <a:latin typeface="Segoe UI Semibold"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Segoe UI"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="CleanFormat">
      <a:fillStyleLst><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="9525"><a:solidFill><a:srgbClr val="CBD5E1"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>"""
            z.writestr("ppt/theme/theme1.xml", theme_xml)

            # 10. Slides and Slide Rels
            for i, (s_xml, s_rels) in enumerate(zip(self.slides_xml, self.slide_rels), start=1):
                z.writestr(f"ppt/slides/slide{i}.xml", s_xml)

                rels_entries = []
                for rel in s_rels:
                    rels_entries.append(
                        f'  <Relationship Id="{rel["id"]}" Type="{rel["type"]}" Target="{rel["target"]}"/>'
                    )
                slide_rels_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{chr(10).join(rels_entries)}
</Relationships>"""
                z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels_xml)

            # 11. Media files
            for m_name, m_data in self.media_files.items():
                z.writestr(f"ppt/media/{m_name}", m_data)

        with open(output_path, "wb") as f:
            f.write(buf.getvalue())
        print(f"Saved {len(self.slides_xml)} slides to {output_path} ({os.path.getsize(output_path)} bytes)")


# ==============================================================================
# Shape and Layout Helpers
# ==============================================================================

class ShapeFactory:
    """Generates valid OpenXML PresentationML shape nodes."""

    def __init__(self) -> None:
        self.shape_counter = 10

    def next_id(self) -> int:
        self.shape_counter += 1
        return self.shape_counter

    def make_card(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        bg_color: Optional[str] = "FFFFFF",
        border_color: Optional[str] = "CBD5E1",
        border_width: int = 12700,
        round_rect: bool = True,
    ) -> str:
        """Draws a container card background with optional rounded corners."""
        s_id = self.next_id()
        geom = "roundRect" if round_rect else "rect"
        fill_xml = (
            f'<a:solidFill><a:srgbClr val="{bg_color}"/></a:solidFill>'
            if bg_color
            else '<a:noFill/>'
        )
        line_xml = (
            f'<a:ln w="{border_width}"><a:solidFill><a:srgbClr val="{border_color}"/></a:solidFill></a:ln>'
            if border_color
            else '<a:ln><a:noFill/></a:ln>'
        )

        return f"""<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{s_id}" name="Card {s_id}"/>
    <p:cNvSpPr/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
    {fill_xml}
    {line_xml}
  </p:spPr>
  <p:txBody>
    <a:bodyPr tIns="0" bIns="0" lIns="0" rIns="0"/>
    <a:lstStyle/>
    <a:p/>
  </p:txBody>
</p:sp>"""

    def make_textbox(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        paragraphs: List[Dict[str, Any]],
        bg_color: Optional[str] = None,
        border_color: Optional[str] = None,
        padding: Tuple[int, int, int, int] = (91440, 91440, 144000, 144000),
        v_align: str = "t",
    ) -> str:
        """Creates a rich text box with multiple formatted paragraphs."""
        s_id = self.next_id()
        top_p, btm_p, left_p, right_p = padding

        fill_xml = (
            f'<a:solidFill><a:srgbClr val="{bg_color}"/></a:solidFill>'
            if bg_color
            else '<a:noFill/>'
        )
        line_xml = (
            f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{border_color}"/></a:solidFill></a:ln>'
            if border_color
            else '<a:ln><a:noFill/></a:ln>'
        )

        p_nodes = []
        for p in paragraphs:
            runs = p.get("runs", [])
            space_after = p.get("space_after", 40000)
            bullet = p.get("bullet", False)
            align = p.get("align", "l")

            p_pr_attrs = f'algn="{align}"'
            p_pr_children = [f'<a:spcAft><a:spcPts val="{int(space_after / 1000)}"/></a:spcAft>']
            if bullet:
                p_pr_children.append('<a:buChar char="•"/>')
            else:
                p_pr_children.append('<a:buNone/>')

            p_pr_xml = f'<a:pPr {p_pr_attrs}>{"".join(p_pr_children)}</a:pPr>'

            run_nodes = []
            for r in runs:
                text = escape_xml(r.get("text", ""))
                size = int(r.get("size", 13) * 100)
                bold = "1" if r.get("bold", False) else "0"
                italic = "1" if r.get("italic", False) else "0"
                color = r.get("color", "0F172A")
                font = r.get("font", "Segoe UI")

                run_nodes.append(f"""<a:r>
  <a:rPr lang="en-GB" sz="{size}" b="{bold}" i="{italic}">
    <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
    <a:latin typeface="{font}"/>
  </a:rPr>
  <a:t>{text}</a:t>
</a:r>""")

            p_nodes.append(f"<a:p>{p_pr_xml}{''.join(run_nodes)}</a:p>")

        return f"""<p:sp>
  <p:nvSpPr>
    <p:cNvPr id="{s_id}" name="TextBox {s_id}"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    {fill_xml}
    {line_xml}
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" rtlCol="0" anchor="{v_align}" tIns="{top_p}" bIns="{btm_p}" lIns="{left_p}" rIns="{right_p}"/>
    <a:lstStyle/>
    {"".join(p_nodes)}
  </p:txBody>
</p:sp>"""

    def make_table(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        headers: List[str],
        rows: List[List[str]],
        col_widths: List[int],
        header_bg: str = "2563EB",
        header_fg: str = "FFFFFF",
        alt_bg: str = "F1F5F9",
        normal_bg: str = "FFFFFF",
        text_fg: str = "0F172A",
        font_size_pt: float = 10.0,
    ) -> str:
        """Constructs an OpenXML DrawingML table with alternating row striping."""
        s_id = self.next_id()
        fs_hundredths = int(font_size_pt * 100)
        h_fs = int((font_size_pt + 0.5) * 100)

        cols_xml = "".join([f'<a:gridCol w="{w}"/>' for w in col_widths])

        # Header Row
        header_cells = []
        for h in headers:
            header_cells.append(f"""<a:tc>
  <a:txBody>
    <a:bodyPr anchor="ctr" tIns="45720" bIns="45720" lIns="91440" rIns="91440"/>
    <a:lstStyle/>
    <a:p>
      <a:pPr algn="l"/>
      <a:r>
        <a:rPr lang="en-GB" sz="{h_fs}" b="1">
          <a:solidFill><a:srgbClr val="{header_fg}"/></a:solidFill>
          <a:latin typeface="Segoe UI Semibold"/>
        </a:rPr>
        <a:t>{escape_xml(h)}</a:t>
      </a:r>
    </a:p>
  </a:txBody>
  <a:tcPr>
    <a:solidFill><a:srgbClr val="{header_bg}"/></a:solidFill>
  </a:tcPr>
</a:tc>""")
        header_row_xml = f'<a:tr h="370000">{"".join(header_cells)}</a:tr>'

        # Body Rows
        body_rows_xml = []
        for r_idx, row in enumerate(rows):
            bg = alt_bg if r_idx % 2 == 1 else normal_bg
            cell_nodes = []
            for c_idx, cell_text in enumerate(row):
                cell_nodes.append(f"""<a:tc>
  <a:txBody>
    <a:bodyPr anchor="ctr" tIns="36000" bIns="36000" lIns="91440" rIns="91440"/>
    <a:lstStyle/>
    <a:p>
      <a:pPr algn="l"/>
      <a:r>
        <a:rPr lang="en-GB" sz="{fs_hundredths}" b="0">
          <a:solidFill><a:srgbClr val="{text_fg}"/></a:solidFill>
          <a:latin typeface="Segoe UI"/>
        </a:rPr>
        <a:t>{escape_xml(cell_text)}</a:t>
      </a:r>
    </a:p>
  </a:txBody>
  <a:tcPr>
    <a:solidFill><a:srgbClr val="{bg}"/></a:solidFill>
    <a:lnB w="6350"><a:solidFill><a:srgbClr val="E2E8F0"/></a:solidFill></a:lnB>
  </a:tcPr>
</a:tc>""")
            body_rows_xml.append(f'<a:tr h="320000">{"".join(cell_nodes)}</a:tr>')

        return f"""<p:graphicFrame>
  <p:nvGraphicFramePr>
    <p:cNvPr id="{s_id}" name="Table {s_id}"/>
    <p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr>
    <p:nvPr/>
  </p:nvGraphicFramePr>
  <p:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></p:xfrm>
  <a:graphic>
    <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">
      <a:tbl>
        <a:tblPr/>
        <a:tblGrid>
          {cols_xml}
        </a:tblGrid>
        {header_row_xml}
        {"".join(body_rows_xml)}
      </a:tbl>
    </a:graphicData>
  </a:graphic>
</p:graphicFrame>"""

    def make_picture(
        self,
        x: int,
        y: int,
        cx: int,
        cy: int,
        rel_id: str,
        border_color: Optional[str] = "CBD5E1",
        border_width: int = 12700,
    ) -> str:
        """Constructs an OpenXML Picture element linking to an embedded image."""
        s_id = self.next_id()
        line_xml = (
            f'<a:ln w="{border_width}"><a:solidFill><a:srgbClr val="{border_color}"/></a:solidFill></a:ln>'
            if border_color
            else ''
        )

        return f"""<p:pic>
  <p:nvPicPr>
    <p:cNvPr id="{s_id}" name="Picture {s_id}"/>
    <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
    <p:nvPr/>
  </p:nvPicPr>
  <p:blipFill>
    <a:blip r:embed="{rel_id}"/>
    <a:stretch><a:fillRect/></a:stretch>
  </p:blipFill>
  <p:spPr>
    <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    {line_xml}
  </p:spPr>
</p:pic>"""


# ==============================================================================
# Slide Decorators (Header, Footer, Badges)
# ==============================================================================

def add_header(
    sf: ShapeFactory,
    shapes: List[str],
    section_num: str,
    title: str,
    subtitle: str,
    is_dark: bool = False,
) -> None:
    """Generates standard top banner and title block."""
    title_fg = "FFFFFF" if is_dark else "0F172A"
    sub_fg = "94A3B8" if is_dark else "64748B"
    badge_bg = "1E293B" if is_dark else "EFF6FF"
    badge_border = "38BDF8" if is_dark else "93C5FD"
    badge_fg = "38BDF8" if is_dark else "2563EB"

    # Category badge
    shapes.append(
        sf.make_textbox(
            x=548640,
            y=280000,
            cx=2200000,
            cy=260000,
            paragraphs=[{
                "runs": [{"text": section_num, "size": 9.5, "bold": True, "color": badge_fg}],
                "space_after": 0,
            }],
            bg_color=badge_bg,
            border_color=badge_border,
            padding=(20000, 20000, 70000, 70000),
            v_align="ctr",
        )
    )

    # Title & Subtitle
    shapes.append(
        sf.make_textbox(
            x=548640,
            y=570000,
            cx=11094720,
            cy=540000,
            paragraphs=[
                {
                    "runs": [{"text": title, "size": 19.0, "bold": True, "color": title_fg}],
                    "space_after": 20000,
                },
                {
                    "runs": [{"text": subtitle, "size": 10.5, "bold": False, "color": sub_fg}],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )


def add_footer(
    sf: ShapeFactory,
    shapes: List[str],
    slide_num: int,
    total_slides: int = 12,
    is_dark: bool = False,
) -> None:
    """Generates standard footer line and attribution."""
    line_fg = "334155" if is_dark else "E2E8F0"
    text_fg = "64748B" if is_dark else "94A3B8"

    # Footer separator line
    shapes.append(sf.make_card(
        x=548640,
        y=6420000,
        cx=11094720,
        cy=15000,
        bg_color=line_fg,
        border_color=None,
        round_rect=False,
    ))

    # Left text
    shapes.append(
        sf.make_textbox(
            x=548640,
            y=6460000,
            cx=6000000,
            cy=280000,
            paragraphs=[{
                "runs": [{
                    "text": "PlangoGraph Retail Planogram Compliance Auditing",
                    "size": 9.0,
                    "color": text_fg,
                }],
                "space_after": 0,
            }],
            padding=(0, 0, 0, 0),
        )
    )

    # Right text
    shapes.append(
        sf.make_textbox(
            x=7000000,
            y=6460000,
            cx=4643360,
            cy=280000,
            paragraphs=[{
                "align": "r",
                "runs": [{
                    "text": f"Slide {slide_num} of {total_slides} | PlangoGraph Evaluation",
                    "size": 9.0,
                    "color": text_fg,
                }],
                "space_after": 0,
            }],
            padding=(0, 0, 0, 0),
        )
    )


# ==============================================================================
# Slide Builders (1 to 12)
# ==============================================================================

def build_slide_1(pb: PresentationBuilder) -> None:
    """Slide 1: Title Slide (Navy Background)."""
    idx = pb.add_slide("0F172A")
    sf = ShapeFactory()
    shapes: List[str] = []

    # Category Pill
    shapes.append(
        sf.make_textbox(
            x=548640,
            y=800000,
            cx=3800000,
            cy=320000,
            paragraphs=[{
                "runs": [{
                    "text": "ACADEMIC YEAR 2026 TO 2027 | B.TECH MAJOR PROJECT DEFENCE",
                    "size": 9.5,
                    "bold": True,
                    "color": "38BDF8",
                }],
                "space_after": 0,
            }],
            bg_color="1E293B",
            border_color="0284C7",
            padding=(40000, 40000, 120000, 120000),
            v_align="ctr",
        )
    )

    # Main Title
    shapes.append(
        sf.make_textbox(
            x=548640,
            y=1260000,
            cx=11094720,
            cy=1400000,
            paragraphs=[
                {
                    "runs": [{
                        "text": "PlangoGraph: Automated Retail Planogram Compliance Auditing via Decoupled ONNX Metric Localisation and 2D Spatial Layout Graphs",
                        "size": 26.0,
                        "bold": True,
                        "color": "FFFFFF",
                    }],
                    "space_after": 80000,
                },
                {
                    "runs": [{
                        "text": "A real-time, CPU-native vision and graph alignment system for fine-grained retail shelf inventory auditing",
                        "size": 13.5,
                        "bold": False,
                        "color": "94A3B8",
                    }],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # Author Card Container
    shapes.append(sf.make_card(
        x=548640,
        y=2850000,
        cx=11094720,
        cy=1950000,
        bg_color="1E293B",
        border_color="334155",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=800000,
            y=3050000,
            cx=5000000,
            cy=1550000,
            paragraphs=[
                {
                    "runs": [
                        {"text": "Investigator & Candidate:\n", "size": 11.0, "color": "38BDF8", "bold": True},
                        {"text": "PlangoGraph Contributors", "size": 16.0, "color": "FFFFFF", "bold": True},
                        {"text": "", "size": 12.5, "color": "CBD5E1"},
                    ],
                    "space_after": 60000,
                },
                {
                    "runs": [
                        {"text": "Department of Computer Science & Engineering\n", "size": 11.5, "color": "E2E8F0"},
                        {"text": "Retail Computer Vision & Graph Alignment\n", "size": 11.5, "color": "E2E8F0"},
                        {"text": "", "size": 11.0, "color": "94A3B8"},
                        {"text": "", "size": 11.0, "color": "38BDF8"},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    shapes.append(
        sf.make_textbox(
            x=6100000,
            y=3050000,
            cx=5200000,
            cy=1550000,
            paragraphs=[
                {
                    "runs": [
                        {"text": "Oral Defence & Evaluation Context:\n", "size": 11.0, "color": "38BDF8", "bold": True},
                        {"text": "External Capstone Project Evaluation (EVAL-2026)\n", "size": 12.5, "color": "FFFFFF", "bold": True},
                    ],
                    "space_after": 60000,
                },
                {
                    "runs": [
                        {"text": "• Core Architecture: ", "size": 11.0, "color": "CBD5E1", "bold": True},
                        {"text": "Decoupled YOLOv8m (SKU-110K) + DINOv2 ViT-S/14 Metric Index\n", "size": 11.0, "color": "94A3B8"},
                        {"text": "• Topological Layer: ", "size": 11.0, "color": "CBD5E1", "bold": True},
                        {"text": "RANSAC Deskewing + Intrinsic Scale Calibration + Spatial NW\n", "size": 11.0, "color": "94A3B8"},
                        {"text": "• Verification Status: ", "size": 11.0, "color": "CBD5E1", "bold": True},
                        {"text": "10/10 Passing Unit Tests | 25.3 FPS Edge CPU Deployment", "size": 11.0, "color": "34D399"},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # 3 High-Impact Stat Badges
    badge_data = [
        ("93.7% mAP@50", "YOLOv8m SKU-110K Localisation (+10.0% vs CVPR'19)"),
        ("100.0% Tilt Recovery", "0 deg to 15 deg Camera Roll Invariance (500 Bays)"),
        ("25.3 FPS Throughput", "39.5 ms End-to-End Latency on Commodity CPU"),
    ]
    b_w = 3450000
    b_gap = 372360
    for b_idx, (val, desc) in enumerate(badge_data):
        bx = 548640 + b_idx * (b_w + b_gap)
        shapes.append(sf.make_card(
            x=bx,
            y=5000000,
            cx=b_w,
            cy=1200000,
            bg_color="1E293B",
            border_color="0284C7",
            round_rect=True,
        ))
        shapes.append(
            sf.make_textbox(
                x=bx + 160000,
                y=5120000,
                cx=b_w - 320000,
                cy=960000,
                paragraphs=[
                    {
                        "runs": [{"text": val, "size": 17.0, "bold": True, "color": "38BDF8"}],
                        "space_after": 30000,
                    },
                    {
                        "runs": [{"text": desc, "size": 10.0, "color": "E2E8F0"}],
                        "space_after": 0,
                    },
                ],
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 1, 12, is_dark=True)
    pb.set_slide_content(idx, "0F172A", "".join(shapes))


def build_slide_2(pb: PresentationBuilder) -> None:
    """Slide 2: Author Details & Project Metadata (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 01",
        "Author Details & Codebase Telemetry",
        "Candidate credentials, institutional affiliation, project domain and structural repository telemetry",
    )

    # Left Column: Author Card
    shapes.append(sf.make_card(
        x=548640,
        y=1300000,
        cx=5200000,
        cy=4900000,
        bg_color="FFFFFF",
        border_color="CBD5E1",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=800000,
            y=1550000,
            cx=4700000,
            cy=4400000,
            paragraphs=[
                {
                    "runs": [{"text": "Investigator Credentials", "size": 15.0, "bold": True, "color": "2563EB"}],
                    "space_after": 80000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Sole Author: ", "bold": True, "color": "0F172A"},
                        {"text": "PlangoGraph Contributors", "color": "0F172A"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Registration Number: ", "bold": True, "color": "0F172A"},
                        {"text": "N/A", "color": "0F172A"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Degree Programme: ", "bold": True, "color": "0F172A"},
                        {"text": "B.Tech in Computer Science & Engineering", "color": "0F172A"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Department: ", "bold": True, "color": "0F172A"},
                        {"text": "Dept. of Computer Science & Engineering", "color": "0F172A"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Institution: ", "bold": True, "color": "0F172A"},
                        {"text": "Open Source Research", "color": "0F172A"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Affiliation: ", "bold": True, "color": "0F172A"},
                        {"text": "Open Source Research", "color": "0F172A"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Location: ", "bold": True, "color": "0F172A"},
                        {"text": "N/A", "color": "0F172A"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Email Contact: ", "bold": True, "color": "0F172A"},
                        {"text": "N/A", "color": "2563EB"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Academic Year: ", "bold": True, "color": "0F172A"},
                        {"text": "2026 to 2027", "color": "0F172A"},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # Right Column: Domain Card (Top)
    shapes.append(sf.make_card(
        x=6000000,
        y=1300000,
        cx=5643360,
        cy=2300000,
        bg_color="FFFFFF",
        border_color="CBD5E1",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=6250000,
            y=1500000,
            cx=5150000,
            cy=1900000,
            paragraphs=[
                {
                    "runs": [{"text": "Project Domain & Technical Paradigm", "size": 15.0, "bold": True, "color": "2563EB"}],
                    "space_after": 60000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Field: ", "bold": True, "color": "0F172A"},
                        {"text": "Retail Computer Vision, Metric Learning and Graph Sequence Alignment", "color": "334155"},
                    ],
                    "space_after": 40000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Core Premise: ", "bold": True, "color": "0F172A"},
                        {"text": "Decoupling dense 1-class packaging localisation from zero-shot metric indexing rather than retraining heavy closed-set detectors", "color": "334155"},
                    ],
                    "space_after": 40000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Execution Model: ", "bold": True, "color": "0F172A"},
                        {"text": "Deploying frozen ONNX graphs on commodity edge CPUs rather than requiring high-latency cloud GPU servers", "color": "334155"},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # Right Column: Codebase Telemetry Card (Bottom)
    shapes.append(sf.make_card(
        x=6000000,
        y=3800000,
        cx=5643360,
        cy=2400000,
        bg_color="EFF6FF",
        border_color="93C5FD",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=6250000,
            y=4000000,
            cx=5150000,
            cy=2000000,
            paragraphs=[
                {
                    "runs": [{"text": "Codebase Telemetry & Verification Status", "size": 15.0, "bold": True, "color": "1E40AF"}],
                    "space_after": 60000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Tracked Source Files: ", "bold": True, "color": "0F172A"},
                        {"text": "17 production modules across core/, api/, and app.py", "color": "1E3A8A"},
                    ],
                    "space_after": 40000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Abstract Syntax Tree Nodes: ", "bold": True, "color": "0F172A"},
                        {"text": "110 AST modular function and class nodes", "color": "1E3A8A"},
                    ],
                    "space_after": 40000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Call-Graph Complexity: ", "bold": True, "color": "0F172A"},
                        {"text": "1,772 directional static and dynamic call-graph edges", "color": "1E3A8A"},
                    ],
                    "space_after": 40000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Automated Unit Tests: ", "bold": True, "color": "0F172A"},
                        {"text": "10/10 passing tests in 2.79 seconds via pytest", "color": "047857", "bold": True},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    add_footer(sf, shapes, 2, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_3(pb: PresentationBuilder) -> None:
    """Slide 3: Problem Statement / Need for the Project (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 02",
        "Problem Statement & Industrial Need",
        "Economic cost of planogram non-compliance and structural failure modes of classical computer vision pipelines",
    )

    # Top Overview Card
    shapes.append(sf.make_card(
        x=548640,
        y=1300000,
        cx=11094720,
        cy=900000,
        bg_color="EFF6FF",
        border_color="93C5FD",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=750000,
            y=1400000,
            cx=10700000,
            cy=700000,
            paragraphs=[
                {
                    "runs": [
                        {"text": "Industrial Context: ", "bold": True, "size": 12.5, "color": "1E40AF"},
                        {"text": "Out-of-stock gaps and misplaced products cause an estimated 4% to 8% loss in retail sales annually. Manual clipboard auditing fails to scale across stores stocking upwards of 10,000 active SKUs, creating an urgent commercial need for automated handheld visual verification.", "size": 11.5, "color": "1E3A8A"},
                    ],
                    "space_after": 0,
                }
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # 4 Failure Mode Cards (2x2 Grid)
    card_w = 5400000
    card_h = 1800000
    c_gap_x = 294720
    c_gap_y = 200000
    grid_y = 2350000

    failures = [
        (
            "Failure Mode 1: Monolithic Multi-Class Churn",
            "Training single-stage multi-class detectors over thousands of SKUs suffers from extreme class imbalance. Whenever product packaging graphics update or seasonal variants rotate, the entire neural network requires hours of GPU retraining rather than onboarding novel items via zero-shot metric embeddings.",
            "DC2626",
            "FEF2F2",
        ),
        (
            "Failure Mode 2: Centroid Roll & Height Fracturing",
            "Clustering detected boxes by vertical centroids (y_c) fractures shelf rows whenever short tins (80 px) sit beside tall cereal bags (180 px). Handheld camera roll up to 15 degrees and double-stacked cans also produce bimodal vertical bands that split single physical shelves into fragmented sub-clusters.",
            "D97706",
            "FFFBEB",
        ),
        (
            "Failure Mode 3: Standoff Distance Scale Distortion",
            "Store auditors capture shelves from standoff distances between 0.8 m and 3.0 m, altering detected product pixel widths drastically. Relying on fixed pixel gap thresholds falsely flags normal item spacing as empty holes on close captures while missing multi-facing gaps on distant photographs.",
            "2563EB",
            "EFF6FF",
        ),
        (
            "Failure Mode 4: String Alignment Ambiguity & Smearing",
            "Classical one-dimensional string edit distance algorithms lack physical coordinates, causing co-occurring deletions and extra insertions to collapse into false substitutions. Concurrently, naive single-image self-calibration copies intruder items into reference blueprints rather than flagging them.",
            "4F46E5",
            "EEF2FF",
        ),
    ]

    for f_idx, (f_title, f_desc, f_accent, f_bg) in enumerate(failures):
        r = f_idx // 2
        c = f_idx % 2
        cx = 548640 + c * (card_w + c_gap_x)
        cy = grid_y + r * (card_h + c_gap_y)

        shapes.append(sf.make_card(
            x=cx,
            y=cy,
            cx=card_w,
            cy=card_h,
            bg_color=f_bg,
            border_color="CBD5E1",
            round_rect=True,
        ))

        shapes.append(
            sf.make_textbox(
                x=cx + 180000,
                y=cy + 160000,
                cx=card_w - 360000,
                cy=card_h - 320000,
                paragraphs=[
                    {
                        "runs": [{"text": f_title, "size": 13.0, "bold": True, "color": f_accent}],
                        "space_after": 40000,
                    },
                    {
                        "runs": [{"text": f_desc, "size": 10.5, "color": "334155"}],
                        "space_after": 0,
                    },
                ],
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 3, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_4(pb: PresentationBuilder) -> None:
    """Slide 4: Project Objectives (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 03",
        "Project Objectives & Architectural Scope",
        "Four primary design objectives formulated to address specific retail computer vision failure modes",
    )

    objectives = [
        (
            "Objective 1: Decoupled 1-Class Localisation and Metric SKU Retrieval",
            "Fine-tune a class-agnostic YOLOv8m detector on 1.73 million boxes from SKU-110K to localise dense packages with high recall. Decouple brand identification into an ONNX-optimised DINOv2 ViT-S/14 metric embedder paired with HSV colour histograms, enabling instant SKU registration without retraining.",
            "0284C7",
        ),
        (
            "Objective 2: Tilt- and Scale-Invariant 2D Spatial Scene Graph Generation",
            "Formulate pairwise bottom-edge slope RANSAC across detected item bases to neutralise camera roll up to 15 degrees. Cluster shelf bottoms via 1D DBSCAN with vertical-span merging, coupled with intrinsic width self-calibration (alpha_k) to maintain scale invariance across 0.8 m to 3.0 m distances.",
            "2563EB",
        ),
        (
            "Objective 3: Spatially-Augmented Needleman-Wunsch Alignment & Graph Edit Distance",
            "Extend global dynamic programming alignment by injecting physical gap tokens directly into the sequence lattice, preventing deletion and insertion collapse. Formulate Markov neighbourhood smoothing during wild calibration to suppress rogue singletons and evaluate compliance via 2D Graph Edit Distance.",
            "4F46E5",
        ),
        (
            "Objective 4: Production CPU Deployment with Modular REST API & Streamlit GUI",
            "Deploy a lightweight runtime executing entirely on commodity CPUs within 200 MB RAM using ONNX Runtime. Implement a FastAPI service layer with Pydantic v2 data models and an interactive 3-tab Streamlit dashboard supporting live camera input, slider tuning and slot diff tables.",
            "059669",
        ),
    ]

    card_y = 1350000
    card_h = 1120000
    card_gap = 130000

    for o_idx, (o_title, o_desc, o_accent) in enumerate(objectives):
        cy = card_y + o_idx * (card_h + card_gap)

        shapes.append(sf.make_card(
            x=548640,
            y=cy,
            cx=11094720,
            cy=card_h,
            bg_color="FFFFFF",
            border_color="CBD5E1",
            round_rect=True,
        ))

        # Accent left border stripe
        shapes.append(sf.make_card(
            x=548640,
            y=cy,
            cx=120000,
            cy=card_h,
            bg_color=o_accent,
            border_color=None,
            round_rect=False,
        ))

        shapes.append(
            sf.make_textbox(
                x=800000,
                y=cy + 140000,
                cx=10600000,
                cy=card_h - 280000,
                paragraphs=[
                    {
                        "runs": [{"text": o_title, "size": 13.5, "bold": True, "color": o_accent}],
                        "space_after": 30000,
                    },
                    {
                        "runs": [{"text": o_desc, "size": 10.5, "color": "334155"}],
                        "space_after": 0,
                    },
                ],
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 4, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_5(pb: PresentationBuilder) -> None:
    """Slide 5: Literature Review - Tabular (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 04",
        "Literature Review: Prior Art in Shelf Auditing",
        "Systematic comparison of foundational methodologies across dense detection, metric retrieval and sequence alignment",
    )

    headers = ["S. No.", "Author(s)", "Year", "Title / Corpus", "Methodology", "Key Findings", "Identified Research Gap"]
    col_widths = [600000, 1400000, 700000, 2000000, 2200000, 2194720, 2000000]

    rows = [
        [
            "1",
            "Merler et al.",
            "2007",
            "GroZi-3.2k In-Situ Recognition",
            "SIFT keypoints and colour histograms matched to studio crops.",
            "Established GroZi-3.2k; exposed in-store illumination shifts.",
            "Cannot segment touching items or model multi-shelf 2D layouts.",
        ],
        [
            "2",
            "Needleman & Wunsch",
            "1970",
            "Global Sequence Alignment",
            "Dynamic programming lattice over 2D scoring matrix.",
            "Guarantees optimal O(nm) global sequence alignment.",
            "Ignores spatial pixel gaps; conflates deletions with extra items.",
        ],
        [
            "3",
            "Goldman et al.",
            "2019",
            "Precise Detection in Dense Scenes",
            "RetinaNet with Soft-IoU and EM-Merger on SKU-110K.",
            "Achieved 49.2% mAP@50-95 on 1.73M densely packed items.",
            "Predicts generic boxes only; lacks SKU recognition and graph checks.",
        ],
        [
            "4",
            "Wei et al.",
            "2019",
            "RPC: Retail Product Checkout",
            "FPN detector with CycleGAN cross-domain synthetic data.",
            "Benchmarked checkout recognition across 200 SKUs.",
            "Closed-set head requires retraining; lacks multi-shelf graph model.",
        ],
        [
            "5",
            "Bai et al.",
            "2020",
            "Products-10K Recognition",
            "ArcFace metric learning margin on 10,000 retail classes.",
            "Separated subtle fine-grained packaging variants at scale.",
            "Operates on isolated crops; provides no structural shelf row grouping.",
        ],
        [
            "6",
            "Oquab et al.",
            "2024",
            "DINOv2 Self-Supervised ViT",
            "Discriminative self-distillation on Vision Transformers.",
            "Extracts universal, lighting-invariant zero-shot descriptors.",
            "Needs chromatic and spatial priors to group crinkled flexible pouches.",
        ],
    ]

    shapes.append(
        sf.make_table(
            x=548640,
            y=1260000,
            cx=11094720,
            cy=3800000,
            headers=headers,
            rows=rows,
            col_widths=col_widths,
            header_bg="2563EB",
            header_fg="FFFFFF",
            alt_bg="F1F5F9",
            normal_bg="FFFFFF",
            text_fg="0F172A",
            font_size_pt=9.5,
        )
    )

    # Bottom Synthesis Banner
    shapes.append(sf.make_card(
        x=548640,
        y=5250000,
        cx=11094720,
        cy=950000,
        bg_color="EFF6FF",
        border_color="93C5FD",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=750000,
            y=5350000,
            cx=10700000,
            cy=750000,
            paragraphs=[
                {
                    "runs": [
                        {"text": "Synthesis and Architectural Implications: ", "bold": True, "size": 11.5, "color": "1E40AF"},
                        {"text": "To bypass the limitations of prior art, localisation must be class-agnostic on SKU-110K (Goldman 2019), SKU identification must use metric embeddings (Oquab 2024) regularised by colour and spatial continuity, and sequence alignment must incorporate physical gap tokens within a 2D scene graph (Needleman & Wunsch 1970).", "size": 10.5, "color": "1E3A8A"},
                    ],
                    "space_after": 0,
                }
            ],
            padding=(0, 0, 0, 0),
        )
    )

    add_footer(sf, shapes, 5, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_6(pb: PresentationBuilder) -> None:
    """Slide 6: System Requirements - Hardware & Software (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 05",
        "System Requirements: Hardware & Software Stack",
        "Asymmetric cloud-edge architectural separation between training and deployment environments",
    )

    # Left Side: Hardware Table
    hw_headers = ["Infrastructure Dimension", "Cloud Training (Google Colab Pro)", "Local Edge Runtime (Workstation)"]
    hw_widths = [1600000, 2000000, 1900000]
    hw_rows = [
        ["Hardware Platform", "Google Colab Pro Environment", "Commodity x86-64 Laptop or Workstation"],
        ["Compute Accelerator", "NVIDIA L4 Tensor Core GPU (22.5 GB)", "None (Pure CPU Execution Provider)"],
        ["System Memory (RAM)", "53.0 GB High-Throughput RAM", "< 200 MB Runtime Memory Footprint"],
        ["Local Storage", "235 GB High-Speed NVMe SSD", "< 500 MB Model & Codebase Footprint"],
        ["Image Acquisition", "SKU-110K Benchmark High-Res Captures", "Standard USB Webcam or Smartphone Photo"],
        ["Operational Goal", "13-Epoch Batch Gradient Descent & Export", "Sub-40 ms Planogram Compliance Auditing"],
    ]

    shapes.append(
        sf.make_table(
            x=548640,
            y=1300000,
            cx=5500000,
            cy=4850000,
            headers=hw_headers,
            rows=hw_rows,
            col_widths=hw_widths,
            header_bg="2563EB",
            header_fg="FFFFFF",
            alt_bg="F8FAFC",
            normal_bg="FFFFFF",
            text_fg="0F172A",
            font_size_pt=9.0,
        )
    )

    # Right Side: Software Stack Cards
    sw_x = 6300000
    sw_w = 5343360
    sw_h = 1100000
    sw_gap = 150000
    sw_start_y = 1300000

    software_components = [
        (
            "Runtime & Execution Engine",
            "Python 3.12+ runtime orchestrating ONNX Runtime >= 1.17.0 on CPUExecutionProvider, delivering fast inference without CUDA dependencies.",
            "2563EB",
        ),
        (
            "Computer Vision & Deep Learning",
            "Ultralytics YOLOv8m (1-class anchor-free detector at 1024px), Meta AI DINOv2 ViT-S/14 (384-d metric embeddings) and OpenCV 4.x for CLAHE deskewing.",
            "0284C7",
        ),
        (
            "Spatial Graph & Dynamic Programming",
            "NetworkX 3.x for attributed directed graph modelling, pairwise slope RANSAC for camera deskewing and custom Needleman-Wunsch DP alignment in NumPy.",
            "4F46E5",
        ),
        (
            "Microservices & Presentation GUI",
            "FastAPI 0.159 with strict Pydantic v2 schema validation for RESTful auditing, accompanied by a 3-tab interactive Streamlit 1.66 dashboard.",
            "059669",
        ),
    ]

    for s_idx, (s_title, s_desc, s_color) in enumerate(software_components):
        sy = sw_start_y + s_idx * (sw_h + sw_gap)

        shapes.append(sf.make_card(
            x=sw_x,
            y=sy,
            cx=sw_w,
            cy=sw_h,
            bg_color="FFFFFF",
            border_color="CBD5E1",
            round_rect=True,
        ))

        shapes.append(sf.make_card(
            x=sw_x,
            y=sy,
            cx=100000,
            cy=sw_h,
            bg_color=s_color,
            border_color=None,
            round_rect=False,
        ))

        shapes.append(
            sf.make_textbox(
                x=sw_x + 180000,
                y=sy + 120000,
                cx=sw_w - 280000,
                cy=sw_h - 240000,
                paragraphs=[
                    {
                        "runs": [{"text": s_title, "size": 12.5, "bold": True, "color": s_color}],
                        "space_after": 20000,
                    },
                    {
                        "runs": [{"text": s_desc, "size": 10.0, "color": "334155"}],
                        "space_after": 0,
                    },
                ],
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 6, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_7(pb: PresentationBuilder) -> None:
    """Slide 7: Proposed Methodology - Architecture & Flowchart (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 06",
        "Proposed Methodology: Four-Stage Architecture",
        "End-to-end mathematical and topological pipeline converting raw store photographs into compliance scorecards",
    )

    stage_w = 2623680
    stage_h = 4850000
    stage_gap = 200000
    start_x = 548640
    start_y = 1300000

    stages = [
        (
            "STAGE 1",
            "Photometric & Geometric Deskewing",
            "core/preprocessing.py",
            [
                ("Input Format: ", "Handheld BGR image I in R^{HxWx3}"),
                ("CIE-LAB CLAHE: ", "Normalises luminance channel L (beta=2.0, 8x8 grid) preserving chromatic hues a, b"),
                ("Pairwise RANSAC: ", "Estimates roll angle theta_ij from bottom-edge slopes across shelf items"),
                ("Quadratic Sag Fit: ", "Fits y_max(x_c) = a*x^2 + b*x + c_k across shelves with >=3 items to remove wide-angle barrel sag"),
            ],
            "2563EB",
        ),
        (
            "STAGE 2",
            "Decoupled Detection & Metric Embeddings",
            "core/detector.py & classifier.py",
            [
                ("Stage 2A Detector: ", "YOLOv8m 1024px ONNX trained on SKU-110K, decoding [1,5,21504] -> [21504,5] with IoU NMS 0.55"),
                ("Contour Fallback: ", "Multi-scale Canny and Sobel seam fallback within [0.0025A, 0.45A]"),
                ("Stage 2B Embedder: ", "DINOv2 ViT-S/14 yields 384-d L2 vector v_i; 24-bin HSV yields colour vector h_i"),
                ("Affinity Metric: ", "A(i,k) = 0.70(v_i*v_k) + 0.30(h_i*h_k) + 0.08*I(SameShelf) >= 0.66"),
            ],
            "0284C7",
        ),
        (
            "STAGE 3",
            "Spatial Scene Graph & Calibration",
            "core/graph_builder.py",
            [
                ("Bottom DBSCAN: ", "Clusters deskewed bases y_max with adaptive radius tau_y = 0.26*h"),
                ("Span Merging: ", "Merges clusters overlapping >=45% vertically to resolve double-stacked cans"),
                ("Intrinsic Scale: ", "alpha_k = clip(median(w_det,k)/median(w_ref,k), 0.15, 6.0) scales reference widths"),
                ("Virtual Gap Nodes: ", "Inserts m_j = round(Delta x / w_cal) __PHYSICAL_GAP__ nodes when Delta x > 0.75*w_cal"),
            ],
            "4F46E5",
        ),
        (
            "STAGE 4",
            "Markov Smoothing & Spatial NW",
            "core/compliance_engine.py",
            [
                ("Markov Blueprint: ", "Filters rogue intrusions [A, B, A] -> [A, A, A] before reference creation"),
                ("Dynamic Matrix: ", "D_ij = max(D_i-1,j-1 + sigma, D_i-1,j + gamma, D_i,j-1 + gamma)"),
                ("Scoring Lattice: ", "Match +2.00, Physical Gap -0.65, Misplaced -1.60, Insertion -1.00"),
                ("2D GED Metric: ", "Evaluates node errors and spatial edge edits to compute compliance score"),
            ],
            "059669",
        ),
    ]

    for s_idx, (s_num, s_title, s_mod, s_bullets, s_accent) in enumerate(stages):
        cx = start_x + s_idx * (stage_w + stage_gap)

        shapes.append(sf.make_card(
            x=cx,
            y=start_y,
            cx=stage_w,
            cy=stage_h,
            bg_color="FFFFFF",
            border_color="CBD5E1",
            round_rect=True,
        ))

        # Top Header Strip
        shapes.append(sf.make_card(
            x=cx,
            y=start_y,
            cx=stage_w,
            cy=500000,
            bg_color=s_accent,
            border_color=None,
            round_rect=False,
        ))

        shapes.append(
            sf.make_textbox(
                x=cx + 120000,
                y=start_y + 80000,
                cx=stage_w - 240000,
                cy=380000,
                paragraphs=[
                    {"runs": [{"text": s_num, "size": 11.0, "bold": True, "color": "FFFFFF"}], "space_after": 10000},
                    {"runs": [{"text": s_title, "size": 10.0, "bold": True, "color": "E0F2FE"}], "space_after": 0},
                ],
                padding=(0, 0, 0, 0),
            )
        )

        # Module sub-badge
        shapes.append(
            sf.make_textbox(
                x=cx + 120000,
                y=start_y + 560000,
                cx=stage_w - 240000,
                cy=220000,
                paragraphs=[{
                    "runs": [{"text": s_mod, "size": 8.5, "bold": True, "color": s_accent}],
                    "space_after": 0,
                }],
                bg_color="F1F5F9",
                border_color="E2E8F0",
                padding=(20000, 20000, 40000, 40000),
                v_align="ctr",
            )
        )

        # Bullets
        bullet_nodes = []
        for b_lbl, b_txt in s_bullets:
            bullet_nodes.append({
                "bullet": True,
                "runs": [
                    {"text": b_lbl, "size": 9.2, "bold": True, "color": "0F172A"},
                    {"text": b_txt, "size": 8.8, "color": "334155"},
                ],
                "space_after": 35000,
            })

        shapes.append(
            sf.make_textbox(
                x=cx + 120000,
                y=start_y + 850000,
                cx=stage_w - 240000,
                cy=stage_h - 950000,
                paragraphs=bullet_nodes,
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 7, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_8(pb: PresentationBuilder) -> None:
    """Slide 8: System Implementation / Modules (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 07",
        "System Implementation: Repository Modules",
        "Structural design contracts, execution entrypoints and automated verification mechanics",
    )

    # Left Side: Module Contract Table
    m_headers = ["Module Path", "Primary Class / Function", "Fault Prevented by Design"]
    m_widths = [1800000, 2100000, 2600000]
    m_rows = [
        ["core/preprocessing.py", "ShelfPreprocessor", "Overhead glare colour shifts; barrel distortion sag."],
        ["core/detector.py", "RetailObjectDetector", "Touching packages merging into single bounding boxes."],
        ["core/classifier.py", "MetricSKUClassifier", "Closed-set model retraining; pouch specularity splits."],
        ["core/graph_builder.py", "SpatialGraphBuilder", "Centroid height row fracturing; standoff scale mismatch."],
        ["core/compliance_engine.py", "PlanogramComplianceEngine", "False substitutions caused by missing and extra items."],
        ["core/visualizer.py", "PlanogramVisualizer", "Black-box scoring by rendering topological edge overlays."],
        ["api/schemas.py", "Pydantic v2 Models", "Malformed JSON, negative box dimensions and type errors."],
        ["app.py & start.sh", "Streamlit & Launcher", "Complex environment setup; provides one-click wild audit."],
    ]

    shapes.append(
        sf.make_table(
            x=548640,
            y=1300000,
            cx=6500000,
            cy=4850000,
            headers=m_headers,
            rows=m_rows,
            col_widths=m_widths,
            header_bg="2563EB",
            header_fg="FFFFFF",
            alt_bg="F8FAFC",
            normal_bg="FFFFFF",
            text_fg="0F172A",
            font_size_pt=8.8,
        )
    )

    # Right Side: Architectural Cards
    rx = 7250000
    rw = 4393360

    right_cards = [
        (
            "AST Call-Graph Telemetry",
            "The repository comprises 17 tracked source files containing 110 AST modular function nodes and 1,772 call-graph edges, ensuring modular separation of concerns.",
            "2563EB",
            "EFF6FF",
        ),
        (
            "Zero-Shot Wild Shelf Onboarding",
            "auto_register_from_detections() extracts crops from unlabelled wild shelf photos, forms SKU clusters via DINOv2+HSV affinity, and exports blueprints without retraining neural weights.",
            "0284C7",
            "F0F9FF",
        ),
        (
            "Deterministic Seam Fallback",
            "If neural weights are absent, a multi-scale Canny and Sobel contour detector activates automatically, preserving operational uptime across edge hardware.",
            "4F46E5",
            "EEF2FF",
        ),
    ]

    rc_y = 1300000
    rc_h = 1500000
    rc_gap = 175000

    for c_idx, (c_title, c_desc, c_accent, c_bg) in enumerate(right_cards):
        cy = rc_y + c_idx * (rc_h + rc_gap)

        shapes.append(sf.make_card(
            x=rx,
            y=cy,
            cx=rw,
            cy=rc_h,
            bg_color=c_bg,
            border_color="CBD5E1",
            round_rect=True,
        ))

        shapes.append(
            sf.make_textbox(
                x=rx + 180000,
                y=cy + 140000,
                cx=rw - 360000,
                cy=rc_h - 280000,
                paragraphs=[
                    {
                        "runs": [{"text": c_title, "size": 13.0, "bold": True, "color": c_accent}],
                        "space_after": 30000,
                    },
                    {
                        "runs": [{"text": c_desc, "size": 10.2, "color": "334155"}],
                        "space_after": 0,
                    },
                ],
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 8, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_9(pb: PresentationBuilder) -> None:
    """Slide 9: Results and Demonstration (Off-White with Embedded Annotated Image)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 08",
        "Experimental Results & System Demonstration",
        "Quantitative benchmarks across training, camera roll, sequence alignment and live annotated shelf audit output",
    )

    # Left Column: Structured Metrics Cards
    lx = 548640
    lw = 5400000

    # Card 1: SKU-110K & Roll Recovery
    shapes.append(sf.make_card(
        x=lx,
        y=1300000,
        cx=lw,
        cy=2300000,
        bg_color="FFFFFF",
        border_color="CBD5E1",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=lx + 180000,
            y=1450000,
            cx=lw - 360000,
            cy=2000000,
            paragraphs=[
                {
                    "runs": [{"text": "Detection & Camera Roll Benchmarks", "size": 14.0, "bold": True, "color": "2563EB"}],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "YOLOv8m SKU-110K Training (Epoch 13): ", "bold": True, "color": "0F172A"},
                        {"text": "Reached 93.7% mAP@50 and 59.2% mAP@50-95 (+10.0% over CVPR'19 baseline).", "color": "334155"},
                    ],
                    "space_after": 35000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Camera Roll Invariance (N=500 Bays): ", "bold": True, "color": "0F172A"},
                        {"text": "Bottom-edge slope RANSAC sustains 100.0% shelf clustering from 0 deg to 15 deg tilt.", "color": "334155"},
                    ],
                    "space_after": 35000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Centroid Baseline Breakdown: ", "bold": True, "color": "0F172A"},
                        {"text": "Naive centroid clustering crashes to 35.1% at 6 deg and 9.0% at 15 deg roll.", "color": "DC2626"},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # Card 2: Anomaly F1 Scores & Latency
    shapes.append(sf.make_card(
        x=lx,
        y=3750000,
        cx=lw,
        cy=2450000,
        bg_color="EFF6FF",
        border_color="93C5FD",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=lx + 180000,
            y=3900000,
            cx=lw - 360000,
            cy=2150000,
            paragraphs=[
                {
                    "runs": [{"text": "Anomaly Detection & Runtime Latency", "size": 14.0, "bold": True, "color": "1E40AF"}],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Out-of-Stock (OOS) Missing Items: ", "bold": True, "color": "0F172A"},
                        {"text": "98.63% Precision, 99.17% Recall, 98.90% F1 score.", "color": "1E3A8A"},
                    ],
                    "space_after": 35000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Misplaced Product Substitutions: ", "bold": True, "color": "0F172A"},
                        {"text": "98.96% Precision, 98.28% Recall, 98.62% F1 score.", "color": "1E3A8A"},
                    ],
                    "space_after": 35000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Extra Product Insertions: ", "bold": True, "color": "0F172A"},
                        {"text": "96.12% F1 score (+15.42 percentage points over 1D string NW).", "color": "047857", "bold": True},
                    ],
                    "space_after": 35000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "End-to-End Latency: ", "bold": True, "color": "0F172A"},
                        {"text": "39.5 ms per bay (25.3 FPS) on commodity CPU with 10/10 passing unit tests.", "color": "1E3A8A"},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # Right Column: Embedded Annotated Shelf Audit Image
    rx = 6200000
    rw = 5443360
    rh = 4082520   # 5443360 * 768 / 1024 to preserve 4:3 aspect ratio exactly

    # Embed media into package
    image_path = "data/outputs/shelf_audit_annotated.png"
    rel_id = pb.add_image_rel(idx, "shelf_audit_annotated.png")
    pb.embed_media("shelf_audit_annotated.png", image_path)

    shapes.append(sf.make_picture(
        x=rx,
        y=1300000,
        cx=rw,
        cy=rh,
        rel_id=rel_id,
        border_color="2563EB",
        border_width=25400,
    ))

    # Image Caption Card
    shapes.append(sf.make_card(
        x=rx,
        y=1300000 + rh + 80000,
        cx=rw,
        cy=720000,
        bg_color="FFFFFF",
        border_color="CBD5E1",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=rx + 150000,
            y=1300000 + rh + 140000,
            cx=rw - 300000,
            cy=600000,
            paragraphs=[{
                "runs": [
                    {"text": "Annotated Audit Output: ", "size": 10.5, "bold": True, "color": "2563EB"},
                    {"text": "Live demonstration showing 86.7% compliance scorecard with green compliant facings, red dashed out-of-stock gaps, amber misplaced items and topological graph edges.", "size": 9.8, "color": "334155"},
                ],
                "space_after": 0,
            }],
            padding=(0, 0, 0, 0),
        )
    )

    add_footer(sf, shapes, 9, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_10(pb: PresentationBuilder) -> None:
    """Slide 10: Conclusion & Scope Boundaries (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 09",
        "Conclusion & Candid System Boundaries",
        "Summary of empirical achievements and deliberate architectural boundaries in retail shelf auditing",
    )

    # Left Column: Achievements Card
    shapes.append(sf.make_card(
        x=548640,
        y=1300000,
        cx=5400000,
        cy=4900000,
        bg_color="FFFFFF",
        border_color="CBD5E1",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=800000,
            y=1550000,
            cx=4900000,
            cy=4400000,
            paragraphs=[
                {
                    "runs": [{"text": "Key Empirical Achievements", "size": 15.0, "bold": True, "color": "2563EB"}],
                    "space_after": 70000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Decoupled Vision Architecture: ", "bold": True, "color": "0F172A"},
                        {"text": "Separating 1-class packaging localisation from zero-shot DINOv2 metric indexing enables new SKU registration in under 20 ms without retraining network weights.", "color": "334155"},
                    ],
                    "space_after": 45000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Geometric Robustness: ", "bold": True, "color": "0F172A"},
                        {"text": "Pairwise slope RANSAC sustains 100.0% shelf clustering across 0 deg to 15 deg camera roll, while intrinsic scale calibration (alpha_k) ensures scale invariance from 0.8 m to 3.0 m standoff distances.", "color": "334155"},
                    ],
                    "space_after": 45000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Spatial Sequence Alignment: ", "bold": True, "color": "0F172A"},
                        {"text": "Injecting spatial gap tokens into Needleman-Wunsch dynamic programming lifted extra-item detection F1 score by 15.42 percentage points over classical string alignment.", "color": "334155"},
                    ],
                    "space_after": 45000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Production Edge Deployment: ", "bold": True, "color": "0F172A"},
                        {"text": "Processes retail bays in 39.5 ms (25.3 FPS) within 200 MB RAM on commodity CPU, backed by 10/10 passing regression unit tests.", "color": "047857", "bold": True},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    # Right Column: Candid Scope Boundaries
    shapes.append(sf.make_card(
        x=6243360,
        y=1300000,
        cx=5400000,
        cy=4900000,
        bg_color="FFFBEB",
        border_color="FCD34D",
        round_rect=True,
    ))

    shapes.append(
        sf.make_textbox(
            x=6500000,
            y=1550000,
            cx=4900000,
            cy=4400000,
            paragraphs=[
                {
                    "runs": [{"text": "Candid Engineering Scope Boundaries", "size": 15.0, "bold": True, "color": "B45309"}],
                    "space_after": 70000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Monocular 2D Front-Facing Scope: ", "bold": True, "color": "0F172A"},
                        {"text": "The pipeline operates on single monocular RGB photographs rather than 3D RGB-D point clouds. It audits front-facing slot compliance and out-of-stock voids, but cannot count units concealed behind the front facing on deep shelves; this is a deliberate boundary of single-view auditing rather than an oversight.", "color": "78350F"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Discrete Shelf Bay Envelope: ", "bold": True, "color": "0F172A"},
                        {"text": "Optimised for discrete retail bays containing up to 30 items per frame. Auditing long continuous supermarket aisles requires multi-frame panorama stitching.", "color": "78350F"},
                    ],
                    "space_after": 50000,
                },
                {
                    "bullet": True,
                    "runs": [
                        {"text": "Static Frozen ONNX Inference: ", "bold": True, "color": "0F172A"},
                        {"text": "Executes frozen neural graphs locally for rapid inference rather than online backpropagation, onboarding new products into the vector catalog dynamically.", "color": "78350F"},
                    ],
                    "space_after": 0,
                },
            ],
            padding=(0, 0, 0, 0),
        )
    )

    add_footer(sf, shapes, 10, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_11(pb: PresentationBuilder) -> None:
    """Slide 11: Future Scope & Roadmap (Off-White)."""
    idx = pb.add_slide("F8FAFC")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "SECTION 10",
        "Future Scope & Research Directions",
        "Architectural extension roadmap spanning multi-frame video stitching, depth estimation and price-rail verification",
    )

    roadmap_items = [
        (
            "Direction 1: Multi-Frame Aisle Panorama Stitching",
            "Continuous Video Homography & Global Scene Graphs",
            [
                ("Homography Tracking: ", "Track feature seams across sequential handheld video frames to register continuous aisle scans."),
                ("Global Scene Graph: ", "Merge discrete bay subgraphs into a unified store-level topological layout graph without perspective distortion."),
                ("Dynamic Motion Compensation: ", "Filter motion blur and auditor walking velocity to maintain packaging detection recall."),
            ],
            "2563EB",
        ),
        (
            "Direction 2: Monocular 3D Volumetric Depth Estimation",
            "Back-of-Shelf Inventory Counting via Depth Foundation Models",
            [
                ("Depth-Anything-V2 Integration: ", "Execute quantized monocular depth estimation to evaluate void depth behind front-facing items."),
                ("Volumetric Stock Estimation: ", "Estimate hidden units on deep pusher-tray shelves, transitioning from 2D slot compliance to volumetric inventory auditing."),
                ("Pusher-Tray Friction Modelling: ", "Model spring tension and packaging geometry to predict stock depletion rates."),
            ],
            "0284C7",
        ),
        (
            "Direction 3: OCR Price-Rail & Barcode Cross-Verification",
            "Multi-Modal Compliance Auditing with Electronic Shelf Labels",
            [
                ("Edge OCR Integration: ", "Incorporate lightweight quantized text recognisers (PaddleOCR / CRNN) to read shelf-edge price labels."),
                ("Price & SKU Cross-Referencing: ", "Verify that detected product brands match the promotional price tags mounted on the shelf rail."),
                ("Barcode Verification: ", "Detect 1D UPC and 2D QR barcodes to validate stock allocation against enterprise inventory databases."),
            ],
            "4F46E5",
        ),
    ]

    card_y = 1350000
    card_h = 1500000
    card_gap = 180000

    for r_idx, (r_title, r_sub, r_bullets, r_color) in enumerate(roadmap_items):
        cy = card_y + r_idx * (card_h + card_gap)

        shapes.append(sf.make_card(
            x=548640,
            y=cy,
            cx=11094720,
            cy=card_h,
            bg_color="FFFFFF",
            border_color="CBD5E1",
            round_rect=True,
        ))

        shapes.append(sf.make_card(
            x=548640,
            y=cy,
            cx=140000,
            cy=card_h,
            bg_color=r_color,
            border_color=None,
            round_rect=False,
        ))

        b_nodes = [
            {"runs": [{"text": f"{r_title} | ", "size": 13.0, "bold": True, "color": r_color}, {"text": r_sub, "size": 11.5, "color": "64748B"}], "space_after": 35000}
        ]
        for b_lbl, b_txt in r_bullets:
            b_nodes.append({
                "bullet": True,
                "runs": [
                    {"text": b_lbl, "size": 10.0, "bold": True, "color": "0F172A"},
                    {"text": b_txt, "size": 9.8, "color": "334155"},
                ],
                "space_after": 25000,
            })

        shapes.append(
            sf.make_textbox(
                x=800000,
                y=cy + 130000,
                cx=10600000,
                cy=card_h - 260000,
                paragraphs=b_nodes,
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 11, 12, is_dark=False)
    pb.set_slide_content(idx, "F8FAFC", "".join(shapes))


def build_slide_12(pb: PresentationBuilder) -> None:
    """Slide 12: References (Navy Background)."""
    idx = pb.add_slide("0F172A")
    sf = ShapeFactory()
    shapes: List[str] = []

    add_header(
        sf,
        shapes,
        "REFERENCES",
        "Foundational References & Literature Citations",
        "Key benchmark publications across dense packaging detection, metric learning and sequence alignment",
        is_dark=True,
    )

    references = [
        (
            "[1]",
            "M. Merler, C. Galleguillos, and S. Belongie",
            "\"Recognizing groceries in situ using in vitro training data,\"",
            "in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2007, pp. 1 to 8.",
        ),
        (
            "[2]",
            "S. B. Needleman and C. D. Wunsch",
            "\"A general method applicable to the search for similarities in the amino acid sequence of two proteins,\"",
            "Journal of Molecular Biology, vol. 48, no. 3, pp. 443 to 453, 1970.",
        ),
        (
            "[3]",
            "E. Goldman, R. Herzig, A. Eisenschtat, J. Goldberger, and T. Hassner",
            "\"Precise detection in densely packed scenes,\"",
            "in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 5227 to 5236.",
        ),
        (
            "[4]",
            "X.-S. Wei, Q. Cui, L. Yang, P. Wang, and L. Liu",
            "\"RPC: A large-scale retail product checkout dataset,\"",
            "arXiv preprint arXiv:1901.07249, 2019.",
        ),
        (
            "[5]",
            "Y. Bai, Y. Chen, W. Yu, L. Wang, and W. Zhang",
            "\"Products-10K: A large-scale product recognition dataset,\"",
            "arXiv preprint arXiv:2008.10545, 2020.",
        ),
        (
            "[6]",
            "M. Oquab et al.",
            "\"DINOv2: Learning robust visual features without supervision,\"",
            "Transactions on Machine Learning Research (TMLR), 2024.",
        ),
    ]

    card_y = 1350000
    card_h = 710000
    card_gap = 120000

    for r_idx, (num, authors, title, venue) in enumerate(references):
        cy = card_y + r_idx * (card_h + card_gap)

        shapes.append(sf.make_card(
            x=548640,
            y=cy,
            cx=11094720,
            cy=card_h,
            bg_color="1E293B",
            border_color="334155",
            round_rect=True,
        ))

        shapes.append(
            sf.make_textbox(
                x=750000,
                y=cy + 80000,
                cx=10700000,
                cy=card_h - 160000,
                paragraphs=[{
                    "runs": [
                        {"text": f"{num}  ", "size": 11.5, "bold": True, "color": "38BDF8"},
                        {"text": f"{authors}, ", "size": 11.0, "bold": True, "color": "FFFFFF"},
                        {"text": f"{title} ", "size": 11.0, "italic": True, "color": "E2E8F0"},
                        {"text": venue, "size": 10.5, "color": "94A3B8"},
                    ],
                    "space_after": 0,
                }],
                padding=(0, 0, 0, 0),
            )
        )

    add_footer(sf, shapes, 12, 12, is_dark=True)
    pb.set_slide_content(idx, "0F172A", "".join(shapes))


# ==============================================================================
# Main Presentation Builder Script
# ==============================================================================

def main() -> None:
    pb = PresentationBuilder()

    print("Building Slide 1: Title Slide (Navy)...")
    build_slide_1(pb)

    print("Building Slide 2: Author Details & Codebase Telemetry...")
    build_slide_2(pb)

    print("Building Slide 3: Problem Statement & Industrial Need...")
    build_slide_3(pb)

    print("Building Slide 4: Project Objectives...")
    build_slide_4(pb)

    print("Building Slide 5: Literature Review (Tabular)...")
    build_slide_5(pb)

    print("Building Slide 6: System Requirements (Hardware & Software)...")
    build_slide_6(pb)

    print("Building Slide 7: Proposed Methodology (Four-Stage Pipeline)...")
    build_slide_7(pb)

    print("Building Slide 8: System Implementation & Repository Modules...")
    build_slide_8(pb)

    print("Building Slide 9: Experimental Results & Live Demonstration...")
    build_slide_9(pb)

    print("Building Slide 10: Conclusion & Scope Boundaries...")
    build_slide_10(pb)

    print("Building Slide 11: Future Scope & Roadmap...")
    build_slide_11(pb)

    print("Building Slide 12: References (Navy)...")
    build_slide_12(pb)

    output_pptx = "docs/plangograph_evaluation_12slides.pptx"
    pb.save(output_pptx)
    print(f"Presentation generated successfully at {output_pptx}")


if __name__ == "__main__":
    main()
