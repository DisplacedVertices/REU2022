#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(1)
import sys, os, argparse, math
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pprint import pprint
from ROOT import TLorentzVector
parser = argparse.ArgumentParser(description = 'folder, file (without .root), bs2derr cut (in cm)')
parser.add_argument('folder', type=str)
parser.add_argument('file', type=str)
parser.add_argument('bs2derr')
args = parser.parse_args()
ROOT.TH1.AddDirectory(False)

# Which plots do you want?
eff_dvv3D = False # Plots efficiency vs dvv3D
deltaphi3D = False # Graphs efficiency vs delta phi3D AND the two delta phi3D graphs overlayed
eff_dbv = False # Plots efficiency vs dbv
eff_dbv3D = False # Plots efficiency vs dbv3D
eff_dvv = False  # Plots efficiency vs dvv
deltaphi = False # Graphs efficiency vs delta phi AND the two delta phi graphs overlayed

recodBV = False # Plots reco dBV before and after cuts

shorter = False # Plots histogram of shorter-lived particle dBV, denominator has 100um cut
short = 0.1 # Xlim for shorter dBV plot
tree = "mfvMiniTreePreSelEvtFilt/t" 

fact2D3D = False # Plots dBV*pi/2 and dBV3D on same graph

bins = 50

folder = args.folder
FILE = args.file
cut = args.bs2derr.split(",")
tau = FILE[FILE.find('tau')+3:FILE.find('tau')+9]
m = FILE[FILE.find('_M0')+3:FILE.find('_M0')+6]
M=m

i = "/nfs/cms/mc1/reu2022/"+folder+"/"+FILE+".root"

if not os.path.isdir(folder):
    os.mkdir(folder)
if not os.path.isdir(folder+"/"+FILE):
    os.mkdir(folder+"/"+FILE)

print "begin " +folder+"/"+FILE+" "+str(cut)

tauint = int(tau)/1000
mint = int(m)
end3D = float(tauint)*2 # Graphs x-range in cm: (0, end) for dvv3D (usually 10 if tau=10mm, 5 if tau=1mm)
end =  float(tauint)*2 #float(tauint)/2 # Graphs x-range in cm: (0, end) for dvv (usually 5 if tau=10mm, 2.5 if tau=1mm)

count = 0
color_list = [0,ROOT.kMagenta,ROOT.kBlue,ROOT.kCyan,ROOT.kRed]

factor_list = []
dvv3D_list = []
dvv_list = []
dbv_list = []
dbv3D_list = []
dphi_list = []
dphi3D_list = []
c0 = ROOT.TCanvas()
c_dvv3D = ROOT.TCanvas()
c_dvv = ROOT.TCanvas()
c_dbv = ROOT.TCanvas()
c_dbv3D = ROOT.TCanvas()
c_phi = ROOT.TCanvas()
c_phi3D = ROOT.TCanvas()
for bs2derr_cut_str in cut:
    count +=1
    color = color_list[count]
    bs2derr_cut = float(bs2derr_cut_str)
    fA = ROOT.TFile.Open(i)
    s = fA.Get(tree)
    if eff_dvv3D:
	h_dvv3D = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV3D truth","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end3D)
	h_dvv3D_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV3D","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end3D)
    if eff_dvv:
	h_dvv = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV truth","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end)
	h_dvv_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end)
    if eff_dbv or fact2D3D: 
	h_dbv = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV truth","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end)
	h_dbv_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end)
    if eff_dbv3D or fact2D3D:
	h_dbv3D = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV3D truth","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end3D)
	h_dbv3D_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV3D","Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); Efficiency", bins, 0, end3D)
    if deltaphi3D or deltaphi:
	h_dp3D_t = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi 3D truth","Bs2derr < "+ str(bs2derr_cut)+"; delta Phi; # Events", bins, 0, 3.15)
	h_dp3D_s = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi 3D selected","Bs2derr < "+ str(bs2derr_cut)+"; delta Phi; Efficiency", bins, 0, 3.15)
	h_dp_t = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi truth","Bs2derr < "+ str(bs2derr_cut)+"; # Events", bins, 0, 3.15)
	h_dp_s = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi selected","Bs2derr < "+ str(bs2derr_cut)+"; delta Phi; Efficiency", bins, 0, 3.15)
    if recodBV:
	h_rdBV = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, Reconstructed dBV before cuts","Reconstructed dBV before cuts; Displacement (cm); # Events", bins, 0, end)
	#h_rdBV_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, Reconstructed dBV after cuts","Reconstructed dBV after cuts; Displacement (cm); # Events", bins, 0, end)
    if fact2D3D:
	h_3Dfactor = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV factor pre cuts","dBV with factor, Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); # Events", bins, 0, end)
	h_3Dfactor_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV factor","dBV with factor, Bs2derr < "+ str(bs2derr_cut)+"; Displacement (cm); # Events", bins, 0, end)
    if shorter:
	h_shorter = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter","dBV of shorter-lived particle before cuts; Displacement (cm); # Events", bins, 0, short)
	h_shorter_cut = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter","dBV of shorter-lived particle after cuts; Displacement (cm); # Events", bins, 0, short)
	h_shorter_pre = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter, before 100um cut","dBV of shorter-lived particle before cuts; Displacement (cm); # Events", bins, 0, short)
	h_shorter_pre_cut = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter, before 100um cut","dBV of shorter-lived particle after cuts; Displacement (cm); # Events", bins, 0, short) 
    for entry in s:
	evt_weight = entry.weight
	evt_run = entry.run
	evt_gen_daughters = entry.gen_daughters
	x = [entry.gen_x[0]+entry.bsx-entry.gen_pv_x0, entry.gen_x[1]+entry.bsx-entry.gen_pv_x0]
	y = [entry.gen_y[0]+entry.bsy-entry.gen_pv_y0, entry.gen_y[1]+entry.bsy-entry.gen_pv_y0]
	z = [entry.gen_z[0]+entry.bsz-entry.gen_pv_z0, entry.gen_z[1]+entry.bsz-entry.gen_pv_z0]
	lfdisp_1 = (x[0]**2 + y[0]**2+z[0]**2)**(0.5)
	lfdisp_2 = (x[1]**2 + y[1]**2+z[1]**2)**(0.5)
	dvv3D = ((x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2)**(0.5)
	dvv = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**(0.5) 
	dphi = abs(entry.gen_lsp_phi[0]-entry.gen_lsp_phi[1])
	dot = (x[0]*x[1]+y[0]*y[1]+z[0]*z[1])/lfdisp_1/lfdisp_2
	dbv1 = (x[0]**2+y[0]**2)**(0.5)
	dbv2 = (x[1]**2+y[1]**2)**(0.5)
	if dot > 0.9999:
	    dphi3D = 0
	else:
	    dphi3D = math.acos(dot)
	if evt_run == 1:
	    if eff_dvv3D:
		h_dvv3D.Fill(dvv3D,evt_weight)
	    if eff_dvv:
		h_dvv.Fill(dvv,evt_weight)
	    if deltaphi:
		h_dp_t.Fill(dphi, evt_weight)
	    if deltaphi3D:
		h_dp3D_t.Fill(dphi3D, evt_weight)
	    if eff_dbv:
		h_dbv.Fill(dbv1,evt_weight)
		h_dbv.Fill(dbv2,evt_weight)
	    if fact2D3D:
		h_3Dfactor.Fill(dbv1/0.5644,evt_weight)
		h_3Dfactor.Fill(dbv2/0.5644,evt_weight)
	    if eff_dbv3D or fact2D3D:
		h_dbv3D.Fill((x[0]**2+y[0]**2+z[0]**2)**(0.5),evt_weight)
		h_dbv3D.Fill((x[1]**2+y[1]**2+z[1]**2)**(0.5),evt_weight)
	    if shorter:
		if dbv1>dbv2:
		    h_shorter_pre.Fill(dbv2,evt_weight)
		    if dbv2 > 0.01:
		        h_shorter.Fill(dbv2,evt_weight)
		else:
		    h_shorter_pre.Fill(dbv1,evt_weight)
		    if dbv1 > 0.01:    
			h_shorter.Fill(dbv1,evt_weight)
	    if eff_dvv or eff_dvv3D or eff_dbv or eff_dbv3D or deltaphi or deltaphi3D or shorter or fact2D3D:
	        xr = []
	        yr = []
	        zr = []
	        phi = []
	        for v in entry.vertices:
		    if v.rescale_bs2derr < bs2derr_cut: 
		        xr.append(v.x) #reconstructed x,y,z coordinates
		        yr.append(v.y)
		        zr.append(v.z)
	        if len(xr)>1:
		    if recodBV:
		        h_rdBV.Fill((xr[0]**2+yr[0]**2)**(0.5))
			h_rdBV.Fill((xr[1]**2+yr[1]**2)**(0.5))
	    	    if eff_dvv3D:
			h_dvv3D_sel.Fill(((x[0]-x[1])**2+(y[0]-y[1])**2+(z[0]-z[1])**2)**(0.5),evt_weight)
	    	    if eff_dvv:
			h_dvv_sel.Fill(((x[0]-x[1])**2+(y[0]-y[1])**2)**(0.5),evt_weight)
	    	    if deltaphi:
			h_dp_s.Fill(dphi, evt_weight)
	    	    if deltaphi3D:
			h_dp3D_s.Fill(dphi3D, evt_weight)
		    if eff_dbv:
			h_dbv_sel.Fill(dbv1, evt_weight)
			h_dbv_sel.Fill(dbv2, evt_weight)
		    if fact2D3D:
			h_3Dfactor_sel.Fill(dbv1*math.pi/2, evt_weight)
			h_3Dfactor_sel.Fill(dbv2*math.pi/2, evt_weight)
	    	    if eff_dbv3D or fact2D3D:
			h_dbv3D_sel.Fill((x[0]**2+y[0]**2+z[0]**2)**(0.5),evt_weight)
			h_dbv3D_sel.Fill((x[1]**2+y[1]**2+z[1]**2)**(0.5),evt_weight)
		    if shorter:
			if dbv1>dbv2:
			    h_shorter_pre_cut.Fill(dbv2,evt_weight)
			    if dbv2>0.01:
			        h_shorter_cut.Fill(dbv2,evt_weight)
			if dbv1<=dbv2:
			    h_shorter_pre_cut.Fill(dbv1,evt_weight)
			    if dbv1>0.01:
			        h_shorter_cut.Fill(dbv1,evt_weight)
    if eff_dvv3D:
	c_dvv3D.cd()
	hratio = h_dvv3D_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dvv3D, 1, 1, "B")
	hratio.SetLineColor(color)
	dvv3D_list.append(hratio.Clone("bs2derr<"+str(bs2derr_cut)))
	ROOT.gStyle.SetOptTitle(0)
    if eff_dvv:
	c_dvv.cd()
	hratio = h_dvv_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dvv, 1, 1, "B")
	hratio.SetLineColor(color)
	dvv_list.append(hratio.Clone("bs2derr<"+str(bs2derr_cut)))
	ROOT.gStyle.SetOptTitle(0)
    if eff_dbv:
	c_dbv.cd()
	hratio = h_dbv_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dbv, 1, 1, "B")
	hratio.SetLineColor(color)
	dbv_list.append(hratio.Clone("bs2derr<"+str(bs2derr_cut)))
	ROOT.gStyle.SetOptTitle(0)
    if fact2D3D:
	c0.cd()
	hratio = h_3Dfactor_sel.Clone('dBV efficiency with factor, bs2derr < '+str(bs2derr_cut))
	hratio.Divide(hratio, h_3Dfactor,1,1,"B")
	hratio.SetLineColor(color-10)
	hratio.SetMaximum(1)
	factor_list.append(hratio.Clone("dBV efficiency with factor, bs2derr < "+str(bs2derr_cut)))
	ROOT.gStyle.SetOptTitle(0)
    if eff_dbv3D or fact2D3D:
	c_dbv3D.cd()
	hratio = h_dbv3D_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dbv3D, 1, 1, "B")
	hratio.SetLineColor(color)
	dbv3D_list.append(hratio.Clone("bs2derr<"+str(bs2derr_cut)))
	ROOT.gStyle.SetOptTitle(0)
    if deltaphi:
	c_phi.cd()
	c_phi.SetLineColor(1)
	h_dp_t.Draw()
	h_dp_s.SetLineColor(color)
	h_dp_s.Draw("Same")
	c0.BuildLegend()
	c0.Print(folder+"/"+FILE+"/dphi_compare_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
	hratio = h_dp_s.Clone("Efficiencies for delta Phi")
	hratio.Divide(hratio, h_dp_t, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dp_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dphi_eff_bs2derr"+str(bs2derr_cut)+".png")
	c_phi.Update()
    if deltaphi3D:
	c_phi3D.cd()
	c_phi3D.SetLineColor(1)
	h_dp3D_t.Draw()
	h_dp3D_s.SetLineColor(color)
	h_dp3D_s.Draw("Same")
	c0.BuildLegend()
	c0.Print(folder+"/"+FILE+"/dphi3D_compare_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
	hratio = h_dp3D_s.Clone("Efficiencies for delta Phi 3D")
	hratio.Divide(hratio, h_dp3D_t, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dp3D_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dphi3D_eff_bs2derr"+str(bs2derr_cut)+".png")
	c_phi3D.Update()
if eff_dvv3D:
    c_dvv3D.cd()
    for i in range(len(dvv3D_list)):
	dvv3D_list[i].Draw('Same')
    c_dvv3D.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dvv3D.Print(folder+"/"+FILE+"/dVV3D_eff_bs2derr.png")
if eff_dvv:
    c_dvv.cd()
    for i in range(len(dvv_list)):
	dvv_list[i].Draw('Same')
    c_dvv.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dvv.Print(folder+"/"+FILE+"/dVV_eff_bs2derr.png")
if eff_dbv:
    c_dbv.cd()
    for i in range(len(dbv_list)):
	dbv_list[i].Draw('Same')
    c_dbv.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dbv.Print(folder+"/"+FILE+"/dBV_eff_bs2derr.png")
if eff_dbv3D:
    c_dbv3D.cd()
    for i in range(len(dbv3D_list)):
	dbv3D_list[i].Draw('Same')
    c_dbv3D.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dbv3D.Print(folder+"/"+FILE+"/dBV3D_eff_bs2derr.png")
if shorter:
    c0.Clear()
    c0.cd()
    h_shorter.Draw()
    h_shorter_cut.SetLineColor(4)
    h_shorter_cut.Draw('Same')
    c0.BuildLegend()
    c0.Print('shorter.png')
    c0.Clear()
    c0.cd()
    hratio = h_shorter_cut.Clone("Shorter dBV efficiency (with vs w/0 100um cut) after 100um cut")
    hratio.Divide(hratio, h_shorter, 1, 1, "B")
    hratio.SetLineColor(color)
    ROOT.gStyle.SetOptTitle(0)
    hratio1 = h_shorter_pre_cut.Clone("Shorter dBV efficiency (with vs w/0 100um cut) before 100um cut")
    hratio1.Divide(hratio1,h_shorter_pre,1,1,"B")
    hratio.SetLineColor(color-10)
    hratio.SetMaximum(1)
    hratio1.Draw()
    hratio.Draw('Same')
    c0.BuildLegend()
    c0.Print(folder+"/"+FILE+"/ratiotest_bs2derr"+str(bs2derr_cut)+".png")
    c0.Clear()
if recodBV:
    c0.cd()
    h_rdBV.Draw()
    c0.Print("rdBV.png")
    c0.Clear()
if fact2D3D:
    c0.cd()
    factor_list[0].Draw()
    dbv3D_list[0].Draw('Same')
    for i in range(1,len(cut)):
	factor_list[i].Draw('Same')
	dbv3D_list[i].Draw('Same')
    c0.BuildLegend()
    c0.Print(folder+"/"+FILE+"/2D3Dfactor.png")
    c0.Clear()
fA.Close()
