#!/usr/bin/env python
import ROOT
import sys, os, argparse, math
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pprint import pprint
from ROOT import TLorentzVector
parser = argparse.ArgumentParser(description = 'Specify tau (in um, 6 digits) and mass (in GeV, 3 digits)')
parser.add_argument('folder', type=str)
parser.add_argument('file', type=str) #no root extension
args = parser.parse_args()

folder = args.folder
FILE = args.file

# Path to input root file
filename_i_list = ["/nfs/cms/mc1/reu2022/"+folder+"/"+FILE+".root"]

# Which plots do you want?
genvertices = False # Plots vertices of gen-level data on 3D histogram
recovertices = False # Plots vertices of reco data on 3D histogram

gentheta = False # Plots distribution of gen-level theta
genphi = False # Plots distribution of gen-level phi
recotheta = False # Plots distribution of reco theta
recophi = False # Plots distribution of reco phi

phirat = False # Plots ratio of phi to phi3D
dphirat = False # Plots ratio of dphi to dphi3D

gendphi = False # Plots distribution of gen-level dphi
gendphi3D = False # Plots distibution of gen-level dphi3D
recodphi = False # Plots distibution of reco dphi
recodphi3D = False # Plots distibution of reco dphi3D

bins = 100

def findall(f, s):
    l = []
    i = -1
    while True:
        i = s.find(f, i+1)
        if i == -1:
            return l
        l.append(s.find(f, i))

tau = FILE[FILE.find('tau')+3:FILE.find('tau')+9]
m = FILE[FILE.find('_M0')+3:FILE.find('_M0')+6]
print tau
print m
tauint = int(tau)/1000
mint = int(m)
axis=tauint/2
M=m

print "begin"

#filename_i_list = options.positional #FIXME : may consider using a wild card to make multiple histograms at once from multiple samples (in *.root files)
for i in filename_i_list:
    first = findall('/', i)[-1]+1
    last = findall('.',i)[-1]
    name = i[first:last]
    fA = ROOT.TFile.Open(i)
    '''if not fA.IsOpen():
         raise IOError('could not open input file %s' % fnA)'''
    t = fA.Get("mfvMiniTree/t") #FIXME : this example opens a tree of a specific event category of >=2vtx >=5trk events (vertices in all event categories have all quality cuts applied except bs2derr)
    if genvertices:
	h_genvertices = ROOT.TH3D("Gen-Level Vertices","Gen-Level Vertices; x; y;z", bins, -axis, axis, bins, -axis, axis, bins, -axis, axis,)
    if recovertices:
	h_recovertices = ROOT.TH3D("Reconstructed Vertices","Reconstructed Vertices; x; y;z", bins, -axis, axis, bins, -axis, axis, bins, -axis, axis,)
    if gentheta:
	h_gentheta = ROOT.TH1F("gentheta","Distribution of Gen-Level Theta; Theta (rad); # Events", bins, 0, 3.15)
    if genphi:
	h_genphi = ROOT.TH1F("genphi","Distribution of Gen-Level Phi; Phi (rad); # Events", bins, -3.15,3.15)
    if recotheta:
	h_recotheta = ROOT.TH1F("recotheta","Distribution of Reconstructed Theta; Theta (rad); # Events", bins, 0, 3.15)
    if recophi:
	h_recophi = ROOT.TH1F("recophi","Distribution of Reconstructed Phi; Phi (rad); # Events", bins, -3.15,3.15)
    if gendphi:
	h_gendphi = ROOT.TH1F("gendphi","Distribution of Gen-Level dPhi; delta Phi (rad); # Events", bins, 0,6.3)
    if gendphi3D:
	h_gendphi3D = ROOT.TH1F("gendphi3D","Distribution of Gen-Level dPhi3D; delta Phi3D (rad); # Events", bins, 0,6.3)
    if recodphi:
	h_recodphi = ROOT.TH1F("recodphi","Distribution of Reconstructed dPhi; delta Phi (rad); # Events", bins, 0,6.3)
    if recodphi3D:
	h_recodphi3D = ROOT.TH1F("recodphi3D","Distribution of Reconstructed dPhi3D; delta Phi3D (rad); # Events", bins, 0,6.3)
    if phirat:
	h_phirat = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, phi","Ratio of Gen-Level phi to phi3D; ratio; # Events", bins, 0, 1.1)
    if dphirat:
	h_dphirat = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dphi","Ratio of Gen-Level dphi to dphi3D; ratio; # Events", bins, 0, 3)
    for entry in t:
        evt_weight = entry.weight
        evt_run = entry.run
        evt_gen_daughters = entry.gen_daughters
	x = [entry.gen_x[0]+entry.bsx-entry.gen_pv_x0, entry.gen_x[1]+entry.bsx-entry.gen_pv_x0]
	y = [entry.gen_y[0]+entry.bsy-entry.gen_pv_y0, entry.gen_y[1]+entry.bsy-entry.gen_pv_y0]
	z = [entry.gen_z[0]+entry.bsz-entry.gen_pv_z0, entry.gen_z[1]+entry.bsz-entry.gen_pv_z0]
        lfdisp_1 = ((x[0])**2 + (y[0])**2+(z[0])**2)**(0.5)
        lfdisp_2 = ((x[1])**2 + (y[1])**2+(z[1])**2)**(0.5)
        dvv3D = (((entry.gen_x[0]+entry.bsx-entry.gen_pv_x0)-(entry.gen_x[1]+entry.bsx-entry.gen_pv_x0))**2 + ((entry.gen_y[0]+entry.bsy-entry.gen_pv_y0)-(entry.gen_y[1]+entry.bsy-entry.gen_pv_y0))**2 + ((entry.gen_z[0]+entry.bsz-entry.gen_pv_z0)-(entry.gen_z[1]+entry.bsz-entry.gen_pv_z0))**2)**(0.5)
        dvv = (((entry.gen_x[0]+entry.bsx-entry.gen_pv_x0)-(entry.gen_x[1]+entry.bsx-entry.gen_pv_x0))**2 + ((entry.gen_y[0]+entry.bsy-entry.gen_pv_y0)-(entry.gen_y[1]+entry.bsy-entry.gen_pv_y0))**2)**(0.5) 
	phival = [math.atan(y[0]/x[0]), math.atan(y[1]/x[1])]
	dot = (x[0]*x[1]+y[0]*y[1]+z[0]*z[1])/lfdisp_1/lfdisp_2
	print 'stored value ' + str(entry.gen_lsp_phi[0])
	print 'calculated value ' + str(math.atan2(y[0],x[0]))
	if dot > 0.9999999:
	    dphi3D = 0
	else:
	    dphi3D = math.acos(dot)
	dot0 = x[0]/lfdisp_1
	if dot0 > 0.9999999:
	    phi3D0 = 0
	else:
	    phi3D0 = math.acos(dot0)
	dot1 = x[1]/lfdisp_2
	if dot1 > 0.9999999:
	    phi3D1 = 0
	else:
	    phi3D1 = math.acos(dot1)
	xr = []
	yr = []
	zr = []
	phi = []
	if phirat:
	    if phi3D0 != 0:
	        h_phirat.Fill(phival[0]/phi3D0, evt_weight)
	    if phi3D1 != 0 :
		h_phirat.Fill(phival[1]/phi3D1, evt_weight)
	if dphirat:
	    if dphi3D != 0:
		h_dphirat.Fill(abs(abs(phival[0])-abs(phival[1]))/dphi3D,evt_weight)
	if gendphi:
	    h_gendphi.Fill(abs(abs(phival[0])-abs(phival[1])),evt_weight)
	if gendphi3D:
	    h_gendphi3D.Fill(dphi3D,evt_weight)
	for v in entry.vertices:
	    xr.append(v.x) #reconstructed x,y,z coordinates
	    yr.append(v.y)
	    zr.append(v.z)
	    phi.append(v.phi)
	if len(xr)>1:
	    mag = [(xr[0]**2+yr[0]**2+zr[0]**2)**(0.5), (xr[1]**2+yr[1]**2+zr[1]**2)**(0.5)]
	    if recovertices:
		h_recovertices.Fill(xr[0],yr[0],zr[0],evt_weight)
		h_recovertices.Fill(xr[1],yr[1],zr[1],evt_weight)
	    if recophi:
		if yr[0]>0:
		    h_recophi.Fill(math.acos(xr[0]/(xr[0]**2+yr[0]**2)**(0.5)),evt_weight)
		else:
		    h_recophi.Fill(-math.acos(xr[0]/(xr[0]**2+yr[0]**2)**(0.5)),evt_weight)
		if yr[0]>0:
		    h_recophi.Fill(math.acos(xr[1]/(xr[1]**2+yr[1]**2)**(0.5)),evt_weight)
		else:
		    h_recophi.Fill(-math.acos(xr[1]/(xr[1]**2+yr[1]**2)**(0.5)),evt_weight)
	    if recotheta:
		h_recotheta.Fill(math.acos(zr[0]/mag[0]),evt_weight)
		h_recotheta.Fill(math.acos(zr[1]/mag[1]),evt_weight)
        if genvertices:
	    h_genvertices.Fill(x[0],y[0],z[0],evt_weight)
	    h_genvertices.Fill(x[1],y[1],z[1],evt_weight)
	if genphi:
	    h_genphi.Fill(entry.gen_lsp_phi[0],evt_weight)
	    h_genphi.Fill(entry.gen_lsp_phi[0],evt_weight)
	if gentheta:
	    h_gentheta.Fill(math.acos(z[0]/lfdisp_1),evt_weight)
	    h_gentheta.Fill(math.acos(z[1]/lfdisp_2) ,evt_weight)
    c0 = ROOT.TCanvas()
    if genvertices:
	h_genvertices.Draw("BOX2")
	c0.Print(folder+"/"+FILE+"/gen3DHist.png")
	c0.Clear()
    if recovertices: 
	h_recovertices.Draw("BOX2")
	c0.Print(folder+"/"+FILE+"/reco3DHist.png")
	c0.Clear()
    if genphi:
	h_genphi.Draw()
	c0.Print(folder+"/"+FILE+"/genphi.png")
	c0.Clear()
    if gentheta:
	h_gentheta.Draw()
	c0.Print(folder+"/"+FILE+"/gentheta.png")
	c0.Clear()
    if recophi:
	h_recophi.Draw()
	c0.Print(folder+"/"+FILE+"/recophi.png")
	c0.Clear()
    if recotheta:
	h_recotheta.Draw()
	c0.Print(folder+"/"+FILE+"/recotheta.png")
	c0.Clear()
    if phirat:
	h_phirat.Draw()
	c0.Print(folder+"/"+FILE+"/phirat.png")
	c0.Clear()
    if dphirat:
	h_dphirat.Draw()
	c0.Print(folder+"/"+FILE+"/dphirat.png")
	c0.Clear()
    fA.Close()
