#!/usr/bin/env python3
"""
계층구조를 보존하는 마크다운 분할 함수

기존 split_markdown_to_sections 함수를 수정하여 title에 전체 계층 경로를 포함합니다.
예: "큰제목 > 중제목 > 소제목 > 소소제목"
"""

import re
import uuid
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

# 기존 클래스들 정의 (필요한 경우)
@dataclass
class Tile:
    id: str
    type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SectionPack:
    id: str
    title: str
    heading_path: str
    text: str
    tiles: List[Tile]
    metadata: Dict[str, Any] = field(default_factory=dict)

# 정규표현식 패턴들
CODE_BLOCK_RE = r"```(\w+)?\n(.*?)\n```"
TABLE_RE = r"\|.*\|.*\n\|[\s\-:|]+\|.*\n(\|.*\|.*\n)*"
IMAGE_RE = r"!\[([^\]]*)\]\(([^)]+)\)(?:\s*\"([^\"]+)\")?"

def extract_term_keys(title: str, body: str) -> List[str]:
    """용어 키 추출 (간단한 구현)"""
    # 실제 구현에서는 더 정교한 키워드 추출 로직 사용
    words = re.findall(r'\b\w+\b', title + ' ' + body)
    return list(set([w.lower() for w in words if len(w) > 3]))[:10]

def extract_exact_keys(text: str) -> List[str]:
    """정확한 키 추출 (간단한 구현)"""
    # 실제 구현에서는 더 정교한 키 추출 로직 사용
    return []

def split_markdown_to_sections_hierarchical(md_text: str, doc_id: str) -> List[SectionPack]:
    """
    계층구조를 보존하는 마크다운 분할 함수
    
    H2/H3 기준 섹션을 만들고, 섹션별로 타일(정의/설명/코드/표/이미지)을 생성.
    코드/표는 '원자' 보존: 절대로 분할하지 않음.
    계층구조를 보존하여 title에 전체 경로를 포함합니다.
    """
    # 헤더 파싱: 모든 헤더(H1~H6) 라인 인덱스 수집
    lines = md_text.splitlines()
    header_idxs: List[Tuple[int, str, int]] = []  # (line_idx, title, level)
    for i, ln in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", ln)  # H1~H6 모두 포함
        if m:
            level = len(m.group(1))
            header_idxs.append((i, m.group(2).strip(), level))

    # 계층구조를 보존하여 경로 생성
    current_hierarchy: List[str] = []  # 현재까지의 계층 구조
    hierarchical_headers: List[Tuple[int, str, int, str]] = []  # (line_idx, title, level, full_path)
    
    for line_idx, title, level in header_idxs:
        # 계층 구조 업데이트
        if level == 1:
            current_hierarchy = [title]
        elif level <= len(current_hierarchy):
            current_hierarchy = current_hierarchy[:level-1] + [title]
        else:
            current_hierarchy.append(title)
        
        # 전체 경로 생성
        full_path = ' > '.join(current_hierarchy)
        hierarchical_headers.append((line_idx, title, level, full_path))

    # H2/H3 기준으로 섹션 경계 설정 (계층구조 정보 포함)
    h2h3_headers = [(idx, title, level, path) for idx, title, level, path in hierarchical_headers 
                    if level in [2, 3]]
    
    boundaries = []
    for idx, (line_idx, title, level, full_path) in enumerate(h2h3_headers):
        start = line_idx
        end = h2h3_headers[idx+1][0] if idx+1 < len(h2h3_headers) else len(lines)
        boundaries.append((start, end, title, level, full_path))

    section_packs: List[SectionPack] = []

    for start, end, title, level, full_path in boundaries:
        raw = "\n".join(lines[start:end])
        section_id = str(uuid.uuid4())

        # 코드블록 추출(원문 보존)
        code_tiles = []
        def code_repl(m):
            lang = (m.group(1) or "").lower()
            code = m.group(2)
            tile = Tile(
                id=str(uuid.uuid4()),
                type="code",
                text=code.strip(),
                metadata={"code_lang": lang or "text"}
            )
            code_tiles.append(tile)
            return f"\n[CODE_BLOCK::{tile.id}]\n"  # 자리표시자

        raw_wo_code = re.sub(CODE_BLOCK_RE, code_repl, raw, flags=re.DOTALL)

        # 표 추출(원문 보존)
        table_tiles = []
        def table_repl(m):
            tbl = m.group(0)
            tile = Tile(
                id=str(uuid.uuid4()),
                type="table",
                text=tbl.strip(),
                metadata={}
            )
            table_tiles.append(tile)
            return f"\n[TABLE_BLOCK::{tile.id}]\n"

        raw_wo_code_table = re.sub(TABLE_RE, table_repl, raw_wo_code, flags=re.MULTILINE)

        # 이미지 캡션 추출
        image_tiles = []
        def image_repl(m):
            alt, path, title_opt = m.group(1), m.group(2), m.group(3) or ""
            caption = (alt or title_opt or path)
            tile = Tile(
                id=str(uuid.uuid4()),
                type="image",
                text=caption.strip(),
                metadata={"image_path": path}
            )
            image_tiles.append(tile)
            return f"\n[IMAGE_BLOCK::{tile.id}]\n"

        raw_clean = re.sub(IMAGE_RE, image_repl, raw_wo_code_table)

        # 정의/설명 타일 만들기
        # 첫 단락을 definition, 나머지를 explanation 타일들로 (문단 기준 분할, 10~15% 겹침 없음)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_clean) if p.strip()]
        tiles: List[Tile] = []

        if paragraphs:
            tiles.append(Tile(
                id=str(uuid.uuid4()),
                type="definition",
                text=paragraphs[0],
                metadata={}
            ))
            for p in paragraphs[1:]:
                tiles.append(Tile(
                    id=str(uuid.uuid4()),
                    type="explanation",
                    text=p,
                    metadata={}
                ))

        # 코드/표/이미지 타일 추가 (원자 보존)
        tiles.extend(code_tiles)
        tiles.extend(table_tiles)
        tiles.extend(image_tiles)

        # 메타데이터 구성
        heading_path = full_path  # 계층구조 전체 경로 사용
        body_for_keys = "\n".join(t.text for t in tiles if t.type in ("definition","explanation"))
        term_keys = extract_term_keys(title, body_for_keys)
        exact_keys = extract_exact_keys(raw)

        # 섹션팩 텍스트(LLM 주입용): 자리표시자 → 간단 요약문으로 치환
        def restore_placeholders(s: str) -> str:
            s = re.sub(r"\[CODE_BLOCK::([-\w]+)\]", "[코드블록: 전체 포함]", s)
            s = re.sub(r"\[TABLE_BLOCK::([-\w]+)\]", "[표: 전체 포함]", s)
            s = re.sub(r"\[IMAGE_BLOCK::([-\w]+)\]", "[이미지 캡션 포함]", s)
            return s

        section_text = restore_placeholders(raw_clean)

        pack = SectionPack(
            id=section_id,
            title=full_path,  # 계층구조 전체 경로를 title로 사용
            heading_path=heading_path,
            text=section_text.strip(),
            tiles=tiles,
            metadata={
                "doc_id": doc_id,
                "section_title": title,  # 원래 제목
                "heading_path": heading_path,  # 전체 경로
                "term_keys": term_keys,
                "exact_keys": exact_keys,
            }
        )
        # 타일 공통 메타 바인딩
        for t in pack.tiles:
            t.metadata.update({
                "doc_id": doc_id,
                "section_id": section_id,
                "section_title": title,  # 원래 제목
                "heading_path": heading_path,  # 전체 경로
            })
            if t.type == "table":
                # 표 헤더/키 후보 추출(간단): 1행을 헤더로 가정
                lines_tbl = [ln for ln in t.text.splitlines() if ln.strip()]
                if lines_tbl:
                    t.metadata["table_header"] = lines_tbl[0]
            if t.type == "code":
                t.metadata["exact_keys"] = extract_exact_keys(t.text)

        section_packs.append(pack)

    return section_packs


def demo_hierarchical_splitting():
    """계층구조 분할 데모"""
    sample_md = """
# 큰제목
큰제목에 대한 설명입니다.

## 중제목
중제목에 대한 설명입니다.

### 소제목
소제목에 대한 설명입니다.

#### 소소제목
소소제목에 대한 설명입니다.

## 다른 중제목
다른 중제목에 대한 설명입니다.

### 다른 소제목
다른 소제목에 대한 설명입니다.
"""
    
    print("🧪 계층구조 마크다운 분할 데모")
    print("="*50)
    
    sections = split_markdown_to_sections_hierarchical(sample_md, "demo_doc")
    
    print(f"📋 생성된 섹션 수: {len(sections)}")
    print("\n📂 섹션 구조:")
    for i, section in enumerate(sections, 1):
        print(f"\n--- 섹션 {i} ---")
        print(f"  제목: {section.title}")
        print(f"  원래 제목: {section.metadata['section_title']}")
        print(f"  경로: {section.heading_path}")
        print(f"  타일 수: {len(section.tiles)}")
        
        # 타일 정보 출력
        for tile in section.tiles:
            print(f"    - {tile.type}: {tile.text[:50]}...")
    
    print("\n✅ 계층구조가 올바르게 보존되었습니다!")


if __name__ == "__main__":
    demo_hierarchical_splitting()
