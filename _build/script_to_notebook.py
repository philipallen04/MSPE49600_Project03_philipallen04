import re, sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook,new_markdown_cell,new_code_cell

src=Path(sys.argv[1] if len(sys.argv)>1 else 'philipallen04_pr03_notebook.py')
out=Path(sys.argv[2] if len(sys.argv)>2 else 'philipallen04_pr03_notebook.ipynb')
lines=src.read_text(encoding='utf-8').splitlines()
# remove shebang/encoding
while lines and (lines[0].startswith('#!') or 'coding:' in lines[0] or lines[0].strip()==''):
    lines.pop(0)
marker_re=re.compile(r'^# In\[\d+\]:\s*$')
markers=[i for i,l in enumerate(lines) if marker_re.match(l)]

def comment_block_to_md(block):
    out=[]
    for line in block:
        if line.startswith('# '): out.append(line[2:])
        elif line=='#': out.append('')
        elif line.startswith('#'): out.append(line[1:].lstrip())
        else: out.append(line)
    while out and not out[0].strip(): out.pop(0)
    while out and not out[-1].strip(): out.pop()
    return '\n'.join(out)

def trim(block):
    while block and not block[0].strip(): block=block[1:]
    while block and not block[-1].strip(): block=block[:-1]
    return block

cells=[]
if markers:
    pre=trim(lines[:markers[0]])
    if pre:
        cells.append(new_markdown_cell(comment_block_to_md(pre)))
for mi,start in enumerate(markers):
    end=markers[mi+1] if mi+1<len(markers) else len(lines)
    region=lines[start+1:end]
    region=trim(region)
    # Find the trailing comment/blank suffix, which nbconvert uses for Markdown cells before next code marker.
    j=len(region)
    while j>0 and (not region[j-1].strip() or region[j-1].lstrip().startswith('#')):
        j-=1
    code_lines=trim(region[:j])
    md_lines=trim(region[j:])
    if code_lines:
        cells.append(new_code_cell('\n'.join(code_lines)))
    if md_lines:
        md=comment_block_to_md(md_lines)
        if md.strip(): cells.append(new_markdown_cell(md))
nb=new_notebook(cells=cells,metadata={
 'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
 'language_info':{'name':'python','version':f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'}
})
nbformat.write(nb,out)
print(f'wrote {out} with {len(cells)} cells')