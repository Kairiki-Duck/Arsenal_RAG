import re

def split_markdown(text, chunk_size=500, overlap=50):
    """
    Split Markdown into logical sections and length-limited chunks.

    Headings start a new logical section. Within each section, paragraphs and
    sentences are kept together where possible, with a character overlap
    between adjacent chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    lines=text.splitlines()
    sections=[]

    current_lines=[]
    heading_stack=[]

    def save_chunk():
        if not current_lines:
            return
        
        content="\n".join(current_lines).strip()

        if not content:
            return
        
        sections.append({
            "text":content,
            "section":" > ".join(heading_stack),
        })
    
    for line in lines:
        match=re.match(r"^(#{1,6})\s+(.+?)\s*$",line)
        if match:
            save_chunk()
            current_lines.clear()
            level=len(match.group(1))
            title=match.group(2)
            heading_stack=heading_stack[:level-1]
            heading_stack.append(title)
        else:
            current_lines.append(line)

    save_chunk()

    chunks=[]
    for section in sections:
        content=section["text"]
        start=0

        while start < len(content):
            remaining=content[start:]
            if len(remaining) <= chunk_size:
                end=len(content)
            else:
                end=start + chunk_size
                boundary_candidates=[
                    remaining.rfind("\n\n", 0, chunk_size + 1),
                    remaining.rfind("\n", 0, chunk_size + 1),
                    remaining.rfind("。", 0, chunk_size + 1),
                    remaining.rfind(".", 0, chunk_size + 1),
                    remaining.rfind("!", 0, chunk_size + 1),
                    remaining.rfind("?", 0, chunk_size + 1),
                    remaining.rfind(" ", 0, chunk_size + 1),
                ]
                boundary=max(boundary_candidates)
                if boundary > chunk_size // 2:
                    end=start + boundary + (1 if remaining[boundary] in ".!?。" else 0)

            chunk_text=content[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text":chunk_text,
                    "section":section["section"],
                })

            if end >= len(content):
                break

            next_start=max(start + 1, end - overlap)
            while next_start < len(content) and content[next_start].isspace():
                next_start += 1
            start=next_start

    return chunks