"""Native MathML helpers shared by the speculative sampling slides."""

def mathbox(x,y,w,h,formula,size=34):
    return f'<foreignObject x="{x}" y="{y}" width="{w}" height="{h}"><div xmlns="http://www.w3.org/1999/xhtml" class="theory-math" style="font-size:{size}px"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{formula}</math></div></foreignObject>'


def mi(s):return f'<mi>{s}</mi>'
def mn(s):return f'<mn>{s}</mn>'
def mo(s):return f'<mo>{s}</mo>'
def row(s):return '<mrow>'+s+'</mrow>'
def frac(a,b):return '<mfrac>'+row(a)+row(b)+'</mfrac>'
def sub(a,b):return '<msub>'+mi(a)+row(b)+'</msub>'
def sup(a,b):return '<msup>'+row(a)+row(b)+'</msup>'
def par(s):return row(mo('(')+s+mo(')'))
def f(name,arg='x'):return mi(name)+par(mi(arg))
def summand(index,term):return '<munder>'+mo('∑')+mi(index)+'</munder>'+term

def finite_sum(index,lower,upper,term):
    return '<munderover>'+mo('∑')+row(mi(index)+mo('=')+mn(lower))+row(upper)+'</munderover>'+term

def minimum():return '<mi mathvariant="normal">min</mi>'+par(f('p')+mo(',')+f('q'))
def positive(arg='x'):return '<msub>'+row(mo('[')+f('p',arg)+mo('−')+f('q',arg)+mo(']'))+mo('+')+'</msub>'
def expectation():return row('<mi mathvariant="normal">E</mi>'+mo('[')+mi('N')+mo(']'))
