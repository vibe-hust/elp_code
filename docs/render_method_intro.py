"""Render a schematic README animation with Pillow (no model execution).

Run: python docs/render_method_intro.py
Requires: Python >= 3.10 and Pillow >= 10. Fonts fall back across macOS/Linux.
The animation illustrates control flow, not measured speed or quality.
"""
from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

W, H, SCALE = 1200, 720, 2
BG = '#FAFBFD'
INK = '#17243B'
MUTED = '#69758A'
LINE = '#DCE3ED'
WHITE = '#FFFFFF'
PALE = '#F0F4FA'
ORANGE = '#C97513'
FONTS = {}


def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in FONTS:
        names = (['/System/Library/Fonts/Menlo.ttc', 'DejaVuSansMono.ttf'] if mono else
                 ['/System/Library/Fonts/Supplemental/Arial Bold.ttf', 'DejaVuSans-Bold.ttf'] if bold else
                 ['/System/Library/Fonts/Supplemental/Arial.ttf', 'DejaVuSans.ttf'])
        for name in names:
            try:
                FONTS[key] = ImageFont.truetype(name, size * SCALE)
                break
            except OSError:
                continue
        else:
            raise RuntimeError('Install DejaVu Sans or edit the font paths in this script.')
    return FONTS[key]


def mix(a, b, t):
    def rgb(c): return tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
    return tuple(round(x + (y-x)*t) for x, y in zip(rgb(a), rgb(b)))


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W*SCALE, H*SCALE), BG)
        self.d = ImageDraw.Draw(self.im)

    def rect(self, box, fill=WHITE, outline=None, width=1, radius=16):
        self.d.rounded_rectangle(tuple(round(v*SCALE) for v in box),
                                 radius=radius*SCALE, fill=fill, outline=outline,
                                 width=width*SCALE)

    def line(self, points, fill=LINE, width=2):
        self.d.line([(round(x*SCALE), round(y*SCALE)) for x, y in points],
                    fill=fill, width=width*SCALE, joint='curve')

    def circle(self, x, y, r, fill, outline=None):
        self.d.ellipse(((x-r)*SCALE, (y-r)*SCALE, (x+r)*SCALE, (y+r)*SCALE),
                       fill=fill, outline=outline, width=2*SCALE)

    def text(self, x, y, value, size=20, color=INK, bold=False, mono=False, center=False):
        f = font(size, bold, mono)
        box = self.d.textbbox((0, 0), value, font=f)
        width = box[2] - box[0]
        px = x*SCALE - (width/2 if center else 0)
        assert px >= -1 and px+width <= W*SCALE+1, (value, px, width)
        self.d.text((px, y*SCALE-box[1]), value, fill=color, font=f)

    def arrow(self, pts, color=LINE, width=3, travel=None):
        self.line(pts, color, width)
        (x0,y0),(x,y)=pts[-2:]
        a=math.atan2(y-y0,x-x0)
        ends=[(x-9*math.cos(a+d), y-9*math.sin(a+d)) for d in [-0.5,0.5]]
        self.line([ends[0],(x,y),ends[1]],color,width)
        if travel is not None:
            lengths=[math.hypot(q[0]-p[0],q[1]-p[1]) for p,q in zip(pts,pts[1:])]
            distance=(travel%1)*sum(lengths)
            for p,q,n in zip(pts,pts[1:],lengths):
                if distance <= n:
                    k=distance/n if n else 0
                    self.circle(p[0]+k*(q[0]-p[0]),p[1]+k*(q[1]-p[1]),6,color)
                    break
                distance-=n

    def chip(self,x,y,w,label,color=ACCENT if 'ACCENT' in globals() else '#2563EB',active=True):
        self.rect((x,y,x+w,y+32),mix(color,WHITE,0.92) if active else PALE,None,radius=8)
        self.text(x+w/2,y+9,label,14,color if active else MUTED,bold=True,center=True)

    def card(self,x,y,w,h,label,active=True):
        self.rect((x,y+3,x+w,y+h+3),'#EBEFF5',None,radius=18)
        self.rect((x,y,x+w,y+h),WHITE,ACCENT if active else LINE,2 if active else 1,18)
        self.text(x+20,y+22,label,19,bold=True)

    def finish(self):
        return self.im.resize((W,H),Image.Resampling.LANCZOS)


def frame(stage,u,elapsed,poster=False):
    c=Canvas()
    c.text(55,30,PROJECT+'  /  METHOD IN MOTION',15,ACCENT,bold=True)
    c.chip(945,23,200,'FROZEN MODEL WEIGHTS',ACCENT)
    c.text(55,73,TITLE,36,bold=True)
    c.text(55,125,SUBTITLE,21,MUTED)
    for i,label in enumerate(STEPS):
        x=55+i*280
        active=i==stage
        c.rect((x,177,x+250,225),mix(ACCENT,WHITE,0.92) if active else WHITE,
               ACCENT if active else LINE,2 if active else 1,12)
        c.circle(x+25,201,12,ACCENT if active else '#E7ECF4')
        c.text(x+25,194,str(i+1),14,WHITE if active else MUTED,bold=True,center=True)
        c.text(x+49,193,label,18,ACCENT if active else MUTED,bold=True)
    draw_method(c,stage,u,poster)
    c.rect((55,597,1145,678),mix(ACCENT,WHITE,0.94),None,radius=14)
    c.text(75,614,CAPTIONS[stage],21,bold=True)
    c.text(75,649,FOOTNOTE,16,MUTED)
    c.rect((55,700,1145,704),'#E7ECF4',None,radius=2)
    c.rect((55,700,55+max(4,1090*elapsed),704),ACCENT,None,radius=2)
    return c.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'assets')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frames=[]
    for n in range(160):
        stage=n//40
        u=(n%40)/39
        frames.append(frame(stage,u,n/159))
    samples=Image.new('RGB',(W,H*4))
    for s in range(4): samples.paste(frames[s*40+25],(0,s*H))
    palette=samples.quantize(colors=192,method=Image.Quantize.MEDIANCUT)
    indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    path=args.output_dir/'method-intro.gif'
    indexed[0].save(path,save_all=True,append_images=indexed[1:],duration=100,loop=0,
                    optimize=True,disposal=1)
    frame(3,1,1,poster=True).save(args.output_dir/'method-overview.png',optimize=True)
    print(f'{PROJECT}: {path} ({path.stat().st_size/1024:.0f} KiB), 1200x720, 16 seconds')


PROJECT='ELP'
ACCENT='#2563EB'
TITLE='Discover how a frozen transformer should loop.'
SUBTITLE='Search over executable programs, then replay the selected computation.'
STEPS=['Propose','Execute','Measure & revise','Freeze & replay']
CAPTIONS=[
 'An agent proposes a loop program: structure, state updates and controls.',
 'The program chooses which blocks to revisit and how to update the state.',
 'Quality, compute cost and runtime diagnostics guide the next candidates.',
 'Replay the saved program at inference, without calling the search agent.',
]
FOOTNOTE='Schematic example of a loop path. Search changes the program; model weights stay fixed.'


def draw_method(c,s,u,poster):
    c.card(55,270,280,246,'Saved program' if s==3 else 'Candidate program',s in [0,3])
    c.text(75,320,'loop(x, evaluate, params)',15,ACCENT,mono=True)
    c.text(75,355,'choose blocks to revisit',16,mono=True)
    c.text(75,385,'update hidden state',16,mono=True)
    c.text(75,415,'return updated state',16,mono=True)
    c.chip(75,461,113,'CODE',ACCENT)
    c.chip(199,461,116,'PARAMETERS',ACCENT)

    c.card(380,270,450,246,'Frozen transformer',s in [1,3])
    c.text(605,322,'Same weights on every pass',16,MUTED,center=True)
    boxes=[(405,370,93,'Blocks'),(549,370,116,'Loop region'),(716,370,90,'Blocks')]
    for i,(x,y,w,label) in enumerate(boxes):
        active=s in [1,3] and (i==1 or poster)
        c.rect((x,y,x+w,y+62),mix(ACCENT,WHITE,0.92) if active else PALE,
               ACCENT if active else LINE,2,10)
        c.text(x+w/2,y+23,label,16,ACCENT if active else INK,bold=True,center=True)
    for a,b in [(498,549),(665,716)]:
        c.arrow([(a,401),(b,401)],ACCENT if s in [1,3] else LINE,travel=u*2 if s in [1,3] else None)
    c.arrow([(646,370),(646,345),(569,345),(569,370)],ACCENT if s in [1,3] else LINE,
            travel=u*3 if s in [1,3] else None)
    c.text(605,465,'Illustrative execution path',15,MUTED,center=True)
    c.arrow([(335,393),(380,393)],ACCENT if s in [0,3] else LINE,
            travel=u if s in [0,3] else None)

    c.card(885,270,260,246,'Model outputs',s in [1,2,3])
    for i,(label,col) in enumerate([('Image','#2563EB'),('Text','#9C50C7'),('Video','#148C83')]):
        y=329+i*53
        c.circle(919,y+10,8,col)
        c.text(942,y,label,20,bold=True)
    c.arrow([(830,393),(885,393)],ACCENT if s in [1,3] else LINE,
            travel=u*2 if s in [1,3] else None)
    col=ACCENT if s==2 or poster else '#B8C3D3'
    c.arrow([(1015,520),(1015,560),(195,560),(195,520)],col,travel=u if s==2 else None)
    c.rect((347,542,865,578),BG,None,radius=0)
    c.text(606,551,'Evaluate quality + cost  →  revise candidates',18,col,bold=True,center=True)


if __name__=='__main__': main()
