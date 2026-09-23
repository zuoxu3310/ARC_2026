#!/usr/bin/env python3
"""Editable story figure. Reads audited results; never fits or changes a model.

Every displayed tile corresponds to a real row of patient_cohort_membership.csv.
PDF, SVG, PNG and native Draw.io objects share one scene specification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'arc_lesion_image_benchmark/results/frozen_prediction_control_2026-09-22'
OUT = ROOT / 'outputs/first-figure-redesign-2026-09-23'
W, H = 7.16 * 72, 438
INK = '#172D3B'
MUTED = '#596D7A'
BLUE = '#286D92'
BLUE_LIGHT = '#EAF3F8'
BLUE_TILE = '#5798B7'
RUST = '#AC542C'
GREEN = '#24786E'
FAINT = '#F2F5F7'
LINE = '#C8D5DD'

plt.rcParams.update({'font.family':'Arial', 'font.size':9, 'text.color':INK,
                     'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
                     'savefig.facecolor':'white', 'figure.facecolor':'white'})


class Scene:
    def __init__(self):
        self.fig = plt.figure(figsize=(W/72, H/72))
        self.ax = self.fig.add_axes([0,0,1,1])
        self.ax.set(xlim=(0,W), ylim=(H,0))
        self.ax.axis('off')
        self.cells, self.edges, self.texts, self.groups = [], [], [], []
        self.serial = 0

    def uid(self, prefix):
        self.serial += 1
        return f'{prefix}_{self.serial:04d}'

    def rect(self,x,y,w,h,fill='white',stroke='none',lw=.6,radius=0,id_=None,parent='1'):
        id_ = id_ or self.uid('shape')
        if radius:
            p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                            facecolor=fill,edgecolor=stroke,linewidth=lw)
        else:
            p=Rectangle((x,y),w,h,facecolor=fill,edgecolor=stroke,linewidth=lw)
        self.ax.add_patch(p)
        self.cells.append(dict(id=id_,kind='rect',x=x,y=y,w=w,h=h,fill=fill,
                               stroke=stroke,lw=lw,radius=radius,parent=parent))
        return id_

    def ellipse(self,x,y,w,h,fill='white',stroke=INK,lw=.7):
        self.ax.add_patch(Ellipse((x+w/2,y+h/2),w,h,facecolor=fill,edgecolor=stroke,linewidth=lw))
        self.cells.append(dict(id=self.uid('ellipse'),kind='ellipse',x=x,y=y,w=w,h=h,
                               fill=fill,stroke=stroke,lw=lw,radius=0,parent='1'))

    def text(self,x,y,t,fs=9,color=INK,bold=False,align='left',width=None,font='Arial'):
        artist=self.ax.text(x,y,t,ha=align,va='center',fontsize=fs,color=color,
                            fontweight='bold' if bold else 'normal',fontfamily=font,
                            linespacing=1.16)
        self.texts.append((artist,width))
        self.cells.append(dict(id=self.uid('text'),kind='text',x=x,y=y,t=t,fs=fs,
                               color=color,bold=bold,align=align,font=font,parent='1'))

    def line(self,points,color=MUTED,lw=.8,arrow=False,dashed=False):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        p=FancyArrowPatch(path=path,arrowstyle='-|>' if arrow else '-',
                          mutation_scale=6,linewidth=lw,color=color,
                          linestyle=(0,(2.5,2)) if dashed else '-',
                          capstyle='round',joinstyle='round')
        self.ax.add_patch(p)
        self.edges.append(dict(id=self.uid('edge'),points=points,color=color,lw=lw,
                               arrow=arrow,dashed=dashed))

    def lock(self,x,y,s=1,color=BLUE):
        self.line([(x+3*s,y+7*s),(x+3*s,y+3*s),(x+5*s,y+s),
                   (x+9*s,y+s),(x+11*s,y+3*s),(x+11*s,y+7*s)],color,1*s)
        self.rect(x,y+7*s,14*s,11*s,fill=color,radius=1.5*s)
        self.ellipse(x+6*s,y+10*s,2*s,2*s,fill='white',stroke='none')
        self.line([(x+7*s,y+12*s),(x+7*s,y+15*s)],'white',.8*s)

    def tile_grid(self,x,y,membership,pitch=4.15,size=3.15,name='grid',color=BLUE_TILE):
        gid=f'group_{name}'
        rows=int(np.ceil(len(membership)/16))
        self.groups.append(dict(id=gid,x=x,y=y,w=16*pitch,h=rows*pitch))
        for i,selected in enumerate(membership):
            fill=color if selected else '#F4F7F9'
            stroke=color if selected else '#D1DCE3'
            self.rect(x+(i%16)*pitch,y+(i//16)*pitch,size,size,
                      fill=fill,stroke=stroke,lw=.22,radius=.4,
                      id_=f'{name}_patient_{i:03d}',parent=gid)

    def save(self,stem):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        text_errors=[]
        for a,maxwidth in self.texts:
            bb=a.get_window_extent(renderer)
            bbpts=bb.transformed(self.ax.transData.inverted())
            if bbpts.x0 < -.2 or bbpts.x1 > W+.2 or min(bbpts.y0,bbpts.y1)<-.2 or max(bbpts.y0,bbpts.y1)>H+.2:
                text_errors.append({'text':a.get_text(),'error':'outside canvas'})
            if maxwidth and abs(bbpts.x1-bbpts.x0)>maxwidth+.5:
                text_errors.append({'text':a.get_text(),'error':'exceeds assigned width',
                                    'actual':abs(bbpts.x1-bbpts.x0),'allowed':maxwidth})
        if text_errors:
            raise ValueError(json.dumps(text_errors,indent=2))
        stem.parent.mkdir(parents=True,exist_ok=True)
        for ext in ['pdf','svg','png']:
            self.fig.savefig(stem.with_suffix('.'+ext),dpi=300)
        self.drawio(stem.with_suffix('.drawio'),renderer)
        return dict(width_in=W/72,height_in=H/72,
                    min_font_pt=min(a.get_fontsize() for a,_ in self.texts),
                    native_vertices=len(self.cells),native_connectors=len(self.edges),
                    native_groups=len(self.groups),text_geometry_errors=text_errors)

    def drawio(self,path,renderer):
        mx=ET.Element('mxfile',host='app.diagrams.net',type='device',version='24.7.17')
        diagram=ET.SubElement(mx,'diagram',name='Figure 1 - fixed-prediction story',id='arc-story')
        model=ET.SubElement(diagram,'mxGraphModel',dx=str(W),dy=str(H),grid='0',
                            guides='1',tooltips='1',connect='1',arrows='1',fold='1',
                            page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),
                            math='0',shadow='0',background='#FFFFFF')
        root=ET.SubElement(model,'root')
        ET.SubElement(root,'mxCell',id='0')
        ET.SubElement(root,'mxCell',id='1',parent='0')
        groups={g['id']:g for g in self.groups}
        emitted_groups=set()
        text_idx=0
        for c in self.cells:
            # Preserve paint order: a grid group follows its background, not precedes it.
            if c['parent']!='1' and c['parent'] not in emitted_groups:
                g=groups[c['parent']]
                group=ET.SubElement(root,'mxCell',id=g['id'],value='',style='group;',vertex='1',parent='1')
                ET.SubElement(group,'mxGeometry',x=str(g['x']),y=str(g['y']),width=str(g['w']),height=str(g['h']),**{'as':'geometry'})
                emitted_groups.add(c['parent'])
            if c['kind']=='text':
                artist,_=self.texts[text_idx]; text_idx+=1
                bb=artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
                # Two points of inset protect native editor text metrics without moving its center.
                w=float(bb.x1-bb.x0)+4; h=abs(float(bb.y1-bb.y0))+3
                x=c['x']-w/2 if c['align']=='center' else c['x']-2 if c['align']=='left' else c['x']-w+2
                y=c['y']-h/2
                style=(f'text;html=0;whiteSpace=wrap;overflow=visible;fillColor=none;strokeColor=none;'
                       f'fontFamily={c["font"]};fontSize={c["fs"]};fontColor={c["color"]};'
                       f'fontStyle={int(c["bold"])};align={c["align"]};verticalAlign=middle;spacing=0;')
                val=c['t']
            else:
                x,y,w,h=c['x'],c['y'],c['w'],c['h']
                if c['parent']!='1':
                    x-=groups[c['parent']]['x']; y-=groups[c['parent']]['y']
                style=(f'{"ellipse" if c["kind"]=="ellipse" else "rounded="+str(int(c["radius"]>0))};'
                       f'fillColor={c["fill"]};strokeColor={c["stroke"]};strokeWidth={c["lw"]};'
                       f'arcSize={2*c["radius"]};absoluteArcSize=1;html=0;shadow=0;')
                val=''
            cell=ET.SubElement(root,'mxCell',id=c['id'],value=val,style=style,vertex='1',parent=c['parent'])
            ET.SubElement(cell,'mxGeometry',x=f'{x:.4f}',y=f'{y:.4f}',width=f'{w:.4f}',height=f'{h:.4f}',**{'as':'geometry'})
        for e in self.edges:
            style=(f'edgeStyle=none;rounded=0;html=0;endArrow={"block" if e["arrow"] else "none"};'
                   f'endFill=1;endSize=4;strokeColor={e["color"]};strokeWidth={e["lw"]};'
                   f'dashed={int(e["dashed"])};')
            cell=ET.SubElement(root,'mxCell',id=e['id'],style=style,edge='1',parent='1')
            geom=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
            for key,pt in [('sourcePoint',e['points'][0]),('targetPoint',e['points'][-1])]:
                ET.SubElement(geom,'mxPoint',x=str(pt[0]),y=str(pt[1]),**{'as':key})
            if len(e['points'])>2:
                arr=ET.SubElement(geom,'Array',**{'as':'points'})
                for pt in e['points'][1:-1]:
                    ET.SubElement(arr,'mxPoint',x=str(pt[0]),y=str(pt[1]))
        ET.indent(mx,space='  ')
        ET.ElementTree(mx).write(path,encoding='utf-8',xml_declaration=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--install',action='store_true')
    args=parser.parse_args()
    members=pd.read_csv(DATA/'patient_cohort_membership.csv')
    # Stable, transparent order; no illustrative selection or invented patients.
    members=members.sort_values(['y_aq','participant_id']).reset_index(drop=True)
    summary=pd.read_csv(DATA/'frozen_prediction_summary.csv')
    rows=summary.query('task == "regression" and model == "RandomForest" and metric == "score"').set_index('subset')
    subsets=['full','has_flair','teghipco_idlist']
    assert len(members)==226
    assert [int(members[k].sum()) for k in subsets]==[226,135,172]
    assert np.allclose([rows.loc[k,'observed'] for k in subsets],
                       [.7123329981819245,.6254958458139793,.7558832622776003],atol=1e-12,rtol=0)
    s=Scene()
    s.text(12,14,'Same predictions. Different patients. Different scores.',12.8,bold=True,width=W-24)
    s.text(12,33,'A controlled test of performance attribution in ARC aphasia prediction',9,color=MUTED,width=W-24)

    # (a) The scientific conflict, condensed into one small causal schematic.
    s.rect(12,48,W-24,47,fill='#F6F8FA',radius=4)
    for j in range(3):
        x,y=23+j*3.5,58+j*3.5
        s.rect(x,y,17,23,fill='white',stroke='#91A8B6',lw=.7,radius=1.5)
        s.ellipse(x+3,y+4,11,13,fill='#E2E9EE',stroke='none')
        s.ellipse(x+4,y+10,4,5,fill=RUST,stroke='none')
    s.text(56,72,'New input\nrequirements',9,bold=True,width=88)
    s.line([(145,72),(158,72),(158,61),(171,61)],color=MUTED,arrow=True)
    s.line([(158,72),(158,84),(171,84)],color=MUTED,arrow=True)
    # Predictor and patients have distinct visual forms, not identical text boxes.
    for x,y in [(180,58),(192,65),(192,53)]:
        s.ellipse(x-2,y-2,4,4,fill=BLUE,stroke='none')
    s.line([(182,58),(190,53)],BLUE,.7)
    s.line([(182,58),(190,65)],BLUE,.7)
    s.text(202,59,'Predictor changes',9,width=113)
    for j in range(4):
        s.ellipse(178+j*5.2,77,3.1,3.1,fill=RUST,stroke='none')
        s.rect(178+j*5.2,81,3.1,5,fill=RUST,radius=.8)
    s.text(202,83,'Eligible patients change',9,width=117)
    s.line([(323,60),(344,60),(344,72),(363,72)],arrow=True)
    s.line([(323,83),(344,83),(344,72)],arrow=False)
    s.text(429,65,'Higher score?',11,bold=True,align='center',width=121)
    s.text(429,82,'How much is method gain?',8.5,color=RUST,align='center',width=131)
    s.text(W/2,105,'(a) The attribution problem',8,font='Times New Roman',align='center')

    # (b) A literal fixed-bank / changing-selection diagram using real membership.
    s.text(12,128,'Our control: freeze predictions, switch the scoring patients',10.8,bold=True,width=W-24)
    s.rect(12,155,133,160,fill=BLUE_LIGHT,stroke='#BAD4E2',lw=.8,radius=5)
    s.rect(12,155,3,160,fill=BLUE)
    s.text(25,171,'Saved predictions',10,bold=True,width=100)
    s.lock(121,162,.9)
    s.text(25,187,'Full-pool held-out values',8.5,color=MUTED,width=112)
    s.tile_grid(45,201,members['full'].to_numpy(),pitch=4.25,size=3.15,name='bank')
    s.text(78.5,278,'226 patients',10,bold=True,align='center',width=116)
    s.text(78.5,300,'Every saved prediction\nstays unchanged',9,color=BLUE,bold=True,align='center',width=116)

    centers=[228,337,450]
    s.line([(146,232),(159,232),(159,150),(450,150)],color=BLUE,lw=1)
    for x in centers:
        s.line([(x,150),(x,165)],color=BLUE,lw=1,arrow=True)
    s.text(312,144,'Change membership only',8.6,color=BLUE,bold=True,align='center',width=228)

    headers=['Full pool','FLAIR available','Published-list\nintersection*']
    colors=[INK,RUST,GREEN]
    records=[]
    for k,cx,label,color in zip(subsets,centers,headers,colors):
        n=int(members[k].sum())
        r=float(rows.loc[k,'observed']); delta=float(rows.loc[k,'delta_from_full'])
        s.text(cx,181,label,9.4,bold=True,align='center',width=100)
        s.text(cx,201,f'n = {n}',8.7,color=MUTED,align='center',width=96)
        s.tile_grid(cx-32.9,213,members[k].to_numpy(),pitch=4.25,size=3.15,name=k)
        s.text(cx,292,f'r = {r:.3f}',18,color=color,bold=True,align='center',width=100)
        s.text(cx,312,'Reference' if k=='full' else f'Change: {delta:+.3f}',9.1,color=color,align='center',width=103)
        records.append(dict(subset=k,n=n,r=r,delta=delta,scope=str(rows.loc[k,'scope'])))

    s.rect(16,328,4,4,fill=BLUE_TILE,stroke=BLUE_TILE,lw=.22)
    s.text(25,330,'Scored',8.3,width=34)
    s.rect(75,328,4,4,fill='#F4F7F9',stroke='#D1DCE3',lw=.3)
    s.text(84,330,'Excluded',8.3,width=44)
    s.text(151,330,'One tile per patient; positions are identical across grids.',8.3,color=MUTED,width=350)
    s.text(12,345,'* Secondary published-list analysis. RF scores are mean Pearson r across 20 repeats.',8.2,color=MUTED,width=W-24)
    s.text(W/2,357,'(b) Score shifts with fixed individual predictions',8,font='Times New Roman',align='center')

    # (c) Locate the control inside the complete argument without recreating a methods checklist.
    s.line([(12,368),(W-12,368)],color=LINE,lw=.65)
    for j,(x,title,detail) in enumerate([
        (12,'Cohort variation is larger','Than model variation in the\ntested competitive set'),
        (181,'Scores shift without refitting','Fixed predictions produce\nboth score gains and losses'),
        (350,'Composition helps explain','The main FLAIR and task-fMRI\nscore losses'),
    ]):
        s.ellipse(x,379,16,16,fill=[BLUE_LIGHT,'#EEF5F3','#FBF2EB'][j],stroke='none')
        s.text(x+8,387,str(j+1),9.4,color=[BLUE,GREEN,RUST][j],bold=True,align='center')
        s.text(x+23,380,title,9.3,bold=True,width=139)
        s.text(x+23,397,detail,8.5,color=MUTED,width=139)
    s.text(W/2,413,'(c) The argument across the study',8,font='Times New Roman',align='center')
    s.text(W/2,430,'A higher score alone does not identify a method gain.',10.5,bold=True,align='center',width=W-24)
    qa=s.save(OUT/'study_design')
    qa.update({'data':records,'patient_order':'Ascending WAB-AQ, then participant_id; same order in all four grids',
               'tile_counts':{k:int(members[k].sum()) for k in subsets},
               'data_sources':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                              for name in ['patient_cohort_membership.csv','frozen_prediction_summary.csv']},
               'source_script':str(Path(__file__).relative_to(ROOT)),
               'raster_generation_used':False,
               'drawio_export':'Native shapes, text, patient-tile groups and connectors; no embedded raster image',
               'claim':'Individual predictions are unchanged; eligibility-based scoring sets move aggregate scores.'})
    page=PdfReader(OUT/'study_design.pdf').pages[0]
    assert abs(float(page.mediabox.width)/72-7.16)<1e-5
    qa['files_sha256']={f'study_design.{ext}':hashlib.sha256((OUT/f'study_design.{ext}').read_bytes()).hexdigest()
                        for ext in ['pdf','svg','png','drawio']}
    (OUT/'qa').mkdir(parents=True, exist_ok=True)
    (OUT/'qa/figure-verification.json').write_text(json.dumps(qa,indent=2)+'\n')
    (OUT/'scene.json').write_text(json.dumps({'width_pt':W,'height_pt':H,'cells':s.cells,'edges':s.edges,'groups':s.groups},indent=2)+'\n')
    if args.install:
        import shutil
        for ext in ['pdf','svg','png','drawio']:
            shutil.copy2(OUT/f'study_design.{ext}',ROOT/f'paper/images/study_design.{ext}')
    print(json.dumps({'figure':str(OUT/'study_design.pdf'),'installed':args.install,**qa},indent=2))


if __name__=='__main__':
    main()
