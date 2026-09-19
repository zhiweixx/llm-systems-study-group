"""Editable SVG teaching diagrams for the Week 3 HTML deck."""
from html import escape as esc
BLUE, INK, MUTED, PALE, LINE = '#245675', '#172329', '#58656d', '#edf3f7', '#aec3d1'
TEAL, WARM = '#397b71', '#a36330'
def text(x, y, value, size=30, weight=400, color=INK, anchor='start', attrs=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {attrs}>{esc(str(value))}</text>'

def lines(x, y, values, size=30, gap=None, **kwargs):
    if isinstance(values, str): values = values.split('\n')
    return ''.join(text(x,y+i*(gap or size*1.4),v,size,**kwargs) for i,v in enumerate(values))

def rect(x,y,w,h,fill='white',stroke=LINE,attrs=''):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" {attrs}/>'

def line(x1,y1,x2,y2,color=LINE,width=1.5,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>'

def arrow(x1,y1,x2,y2,color=BLUE):
    from math import hypot
    length = hypot(x2-x1, y2-y1)
    ux, uy = (x2-x1)/length, (y2-y1)/length
    ax, ay = x2-8*ux+5*uy, y2-8*uy-5*ux
    bx, by = x2-8*ux-5*uy, y2-8*uy+5*ux
    return line(x1,y1,x2,y2,color,2)+f'<path d="M {ax} {ay} L {x2} {y2} L {bx} {by}" fill="none" stroke="{color}" stroke-width="2"/>'

def label_box(x,y,w,h,label,fill=PALE,size=28):
    return rect(x,y,w,h,fill)+text(x+w/2,y+h/2+size*.34,label,size,600,anchor='middle')

def takeaway(value, second=None):
    s=line(75,740,1525,740)+text(75,785,value,30,700,color=BLUE)
    if second: s+=text(75,820,second,23,color=MUTED)
    return s

def control(kind,x=75,y=667,extra=''):
    return f'''<foreignObject x="{x}" y="{y}" width="1450" height="64"><div xmlns="http://www.w3.org/1999/xhtml" class="interaction" data-deck-ignore-keys="true"><button type="button" data-step-prev="{kind}">Previous step</button><button type="button" data-step-next="{kind}">Next step</button><span data-step-label="{kind}" aria-live="polite"></span>{extra}</div></foreignObject>'''

def table(x,y,width,headers,rows,ratios=None,row_h=74,size=28):
    ratios=ratios or [1/len(headers)]*len(headers)
    xs=[x]; out=rect(x,y,width,row_h,PALE,'none')
    for r in ratios: xs.append(xs[-1]+width*r)
    for i,h in enumerate(headers): out+=text(xs[i]+14,y+46,h,size,700)
    out+=line(x,y+row_h,x+width,y+row_h)
    for j,row in enumerate(rows):
        yy=y+(j+1)*row_h
        for i,v in enumerate(row): out+=text(xs[i]+14,yy+46,v,size)
        out+=line(x,yy+row_h,x+width,yy+row_h,'#c7cdd1')
    return out

def matrix(x,y,name,rows=4,cols=4,cell=44,group='',values=None):
    out=text(x,y-17,name,27,700)
    for r in range(rows):
        for c in range(cols):
            out+=f'<g data-matrix="{group}" data-row="{r}" data-col="{c}">'+rect(x+c*cell,y+r*cell,cell,cell,'white')
            if values is not None: out+=text(x+(c+.5)*cell,y+(r+.67)*cell,values[r][c],23,anchor='middle')
            out+='</g>'
    return out


def slide(title, body, notes, sources=(), section='', **kwargs):
    return dict(title=title, body=body, notes=notes, sources=list(sources), section=section, **kwargs)

def code(x, y, w, h, value, size=26):
    return f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h}"><pre xmlns="http://www.w3.org/1999/xhtml" class="code-block" style="font-size:{size}px">{esc(value)}</pre></foreignObject>'

def paragraph(x, y, w, h, value, size=28):
    return f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h}"><div xmlns="http://www.w3.org/1999/xhtml" class="prose" style="font-size:{size}px">{value}</div></foreignObject>'

def steps(kind, scenes, x=75, y=666):
    return ''.join(f'<g data-sequence="{kind}" data-frame="{i}" style="display:{"inline" if i==0 else "none"}">{scene}</g>' for i,scene in enumerate(scenes))+control(kind,x,y)
