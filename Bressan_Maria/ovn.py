#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(1)
import sys, os, argparse, math
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pprint import pprint
from ROOT import TLorentzVector
parser = argparse.ArgumentParser(description = 'file (without .root)')
parser.add_argument('oldfolder', type=str)
parser.add_argument('newfolder', type=str)
parser.add_argument('file', type=str)
parser.add_argument('tau', type=str) #in um
parser.add_argument('mass', type=str)
args = parser.parse_args()
ROOT.TH1.AddDirectory(False)

# Old vs New: Which plots do you want?
eff_dvv3D = False # Plots efficiency vs dvv3D
deltaphi3D = False # Graphs efficiency vs delta phi3D
eff_dbv = False # Plots efficiency vs dbv
eff_dbv3D = False # Plots efficiency vs dbv3D
eff_dvv = False  # Plots efficiency vs dvv
deltaphi = False # Graphs efficiency vs delta phi 
eff_err = True # Plots efficiency vs bs2derr cut

nsvcel = False # Plots number of vertices

shorter = False # Plots histogram of shorter-lived particle dBV, denominator has 100um cut
short = 0.1 # Xlim for shorter dBV plot
tree = "mfvMiniTreePreSelEvtFilt/t" 
recodBV=False
fact2D3D = False # Plots dBV*pi/2 and dBV3D on same graph

recodVV = True #Overlays old and new reconstructed dVV

bins = 30

oldfolder = args.oldfolder
newfolder = args.newfolder
folder = [oldfolder,newfolder]
legend = ['old', 'new']
phibins = 50
endphi = 3.15
FILE = args.file
bs2derr_cut_str = 0.0025
tau = args.tau
m = args.mass
M=m

files = ["/nfs/cms/mc1/reu2022/"+folder[0]+"/"+FILE+".root","/nfs/cms/mc1/reu2022/"+folder[1]+"/"+FILE+".root"]

if not os.path.isdir(folder[-1]):
    os.mkdir(folder[-1])
if not os.path.isdir(folder[-1]+"/"+FILE):
    os.mkdir(folder[-1]+"/"+FILE)

print "begin " +folder[-1]+"/"+FILE+" "+str(bs2derr_cut_str)

tauint = float(tau)/1000
mint = int(m)
end3D = tauint/2 #float(tauint)/2 # Graphs x-range in cm: (0, end) for dvv3D (usually 10 if tau=10mm, 5 if tau=1mm)
end =  tauint/2 #float(tauint)/2 #float(tauint)/2 # Graphs x-range in cm: (0, end) for dvv (usually 5 if tau=10mm, 2.5 if tau=1mm)

marker = [4,5]

count = 0
color_list = [0,ROOT.kMagenta,ROOT.kBlue,ROOT.kCyan,ROOT.kRed]
integral = []
factor_list = []
h_err_list = []
dvv3D_list = []
dvv_list = []
dbv_list = []
dbv3D_list = []
dphi_list = []
recodVV_list = []
dphi3D_list = []
nv_list = []
c0 = ROOT.TCanvas()
c_dvv3D = ROOT.TCanvas()
c_dvv = ROOT.TCanvas()
c_dbv = ROOT.TCanvas()
c_dbv3D = ROOT.TCanvas()
c_phi = ROOT.TCanvas()
c_phi3D = ROOT.TCanvas()
for i in files:
    count +=1
    color = color_list[count]
    bs2derr_cut = float(bs2derr_cut_str)
    fA = ROOT.TFile.Open(i)
    s = fA.Get(tree)
    if eff_err:
	h_err = ROOT.TH2F("bs2derr","Bs2derr; bs2derr of 1st vertex (cm); bs2derr of 2nd vertex (cm)", bins,0,0.01,bins,0,0.01)
    if eff_dvv3D:
	h_dvv3D = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV3D truth", legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end3D)
	h_dvv3D_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV3D",legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end3D)
    if eff_dvv:
	h_dvv = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV truth",legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end)
	h_dvv_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV",legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end)
    if eff_dbv or fact2D3D: 
	h_dbv = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV truth",legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end)
	h_dbv_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV",legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end)
    if eff_dbv3D or fact2D3D:
	h_dbv3D = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV3D truth",legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end3D)
	h_dbv3D_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV3D",legend[count-1]+"; Displacement (cm); Efficiency", bins, 0, end3D)
    if deltaphi3D or deltaphi:
	h_dp3D_t = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi 3D truth",legend[count-1]+"; delta Phi; # Events", phibins, 0, endphi)
	h_dp3D_s = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi 3D selected",legend[count-1]+"; delta Phi; Efficiency", phibins, 0, endphi)
	h_dp_t = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi truth",legend[count-1]+"; # Events", phibins, 0, endphi)
	h_dp_s = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi selected",legend[count-1]+"; delta Phi; Efficiency", phibins, 0, endphi)
    if fact2D3D:
	h_3Dfactor = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV factor pre cuts","dBV with factor, " +legend[count-1]+"; Displacement (cm); # Events", bins, 0, end)
	h_3Dfactor_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV factor","dBV with factor, "+legend[count-1]+"; Displacement (cm); # Events", bins, 0, end)
    if shorter:
	h_shorter = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter","dBV of shorter-lived particle before cuts; Displacement (cm); # Events", bins, 0, short)
	h_shorter_cut = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter","dBV of shorter-lived particle after cuts; Displacement (cm); # Events", bins, 0, short)
	h_shorter_pre = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter, before 100um cut","dBV of shorter-lived particle before cuts; Displacement (cm); # Events", bins, 0, short)
	h_shorter_pre_cut = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, shorter, before 100um cut","dBV of shorter-lived particle after cuts; Displacement (cm); # Events", bins, 0, short) 
    if recodVV:
	h_recodVV = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, recodVV",legend[count-1]+"; dVV (cm); # Events", bins, 0, end)
    if nsvcel:
	h_nv = ROOT.TH1F("nv",legend[count-1]+"; # vertices; # Events", 5, 0, 5)
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
	    if True:
	        xr = []
	        yr = []
	        zr = []
	        phi = []
	        for v in entry.vertices:
		    if v.rescale_bs2derr < bs2derr_cut: 
		        xr.append(v.x) #reconstructed x,y,z coordinates
		        yr.append(v.y)
		        zr.append(v.z)
		if nsvcel:
		    h_nv.Fill(len(xr),evt_weight)
	        if len(xr)>1:
		    if eff_err:
			err_list = []
			for v in entry.vertices:
		    	    err_list.append(v.bs2derr)
			if len(err_list)>1:
		    	    h_err.Fill(err_list[0],err_list[1],evt_weight)
		    if recodVV:
			h_recodVV.Fill((xr[0]**2+yr[0]**2)**(0.5),evt_weight)
		    if recodBV:
		        h_rdBV.Fill((xr[0]**2+yr[0]**2)**(0.5),evt_weight)
			h_rdBV.Fill((xr[1]**2+yr[1]**2)**(0.5),evt_weight)
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
		    if eff_err:
			h_err_list.append(h_err)
    if nsvcel:
	c0.cd()
	h_nv.SetLineColor(color)
	h_nv.SetMarkerStyle(marker[count-1])
	h_nv.SetMarkerColor(color)
	nv_list.append(h_nv.Clone())
    if eff_dvv3D:
	c_dvv3D.cd()
	hratio = h_dvv3D_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dvv3D, 1, 1, "B")
	hratio.SetLineColor(color)
	hratio.SetMarkerStyle(marker[count-1])
	hratio.SetMarkerColor(color)
	dvv3D_list.append(hratio.Clone())
	ROOT.gStyle.SetOptTitle(0)
    if eff_dvv:
	c_dvv.cd()
	hratio = h_dvv_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dvv, 1, 1, "B")
	hratio.SetLineColor(color)
	hratio.SetMarkerStyle(marker[count-1])
	hratio.SetMarkerColor(color)
	dvv_list.append(hratio.Clone())
	ROOT.gStyle.SetOptTitle(0)
    if eff_dbv:
	c_dbv.cd()
	hratio = h_dbv_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dbv, 1, 1, "B")
	hratio.SetLineColor(color)
	hratio.SetMarkerStyle(marker[count-1])
	hratio.SetMarkerColor(color)
	dbv_list.append(hratio.Clone())
	ROOT.gStyle.SetOptTitle(0)
    if fact2D3D:
	c0.cd()
	hratio = h_3Dfactor_sel.Clone('dBV efficiency with factor, bs2derr < '+str(bs2derr_cut))
	hratio.Divide(hratio, h_3Dfactor,1,1,"B")
	hratio.SetLineColor(color-10)
	hratio.SetMaximum(1)
	hratio.SetMarkerStyle(marker[count-1])
	hratio.SetMarkerColor(color)
	factor_list.append(hratio.Clone("dBV efficiency with factor, bs2derr < "+str(bs2derr_cut)))
	ROOT.gStyle.SetOptTitle(0)
    if eff_dbv3D or fact2D3D:
	c_dbv3D.cd()
	hratio = h_dbv3D_sel.Clone("Efficiencies for dvv3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dbv3D, 1, 1, "B")
	hratio.SetLineColor(color)
	hratio.SetMarkerStyle(marker[count-1])
	hratio.SetMarkerColor(color)
	dbv3D_list.append(hratio.Clone())
	ROOT.gStyle.SetOptTitle(0)
    if deltaphi:
	c_phi.cd()
	hratio = h_dp_s.Clone("Efficiencies for dPhi, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dp_t, 1, 1, "B")
	hratio.SetLineColor(color)
	hratio.SetMarkerStyle(marker[count-1])
	hratio.SetMarkerColor(color)
	dphi_list.append(hratio.Clone())
	ROOT.gStyle.SetOptTitle(0)
    if deltaphi3D:
	c_phi3D.cd()
	hratio = h_dp3D_s.Clone("Efficiencies for dPhi3D, bs2derr < "+str(bs2derr_cut))
	hratio.Divide(hratio, h_dp3D_t, 1, 1, "B")
	hratio.SetLineColor(color)
	hratio.SetMarkerStyle(marker[count-1])
	hratio.SetMarkerColor(color) 
	dphi3D_list.append(hratio.Clone())
	ROOT.gStyle.SetOptTitle(0)
    if recodVV:
	c0.cd()
	h_recodVV.SetLineColor(color)
	h_recodVV.SetMarkerStyle(marker[count-1])
	h_recodVV.SetMarkerColor(color)
	ROOT.gStyle.SetOptTitle(0)
	recodVV_list.append(h_recodVV.Clone())
	integral.append(h_recodVV.Integral(0,bins))
	mean = h_recodVV.GetMean()
    if eff_err:
	cut_array = np.linspace(0,0.01,bins+1)
	err_array0 = [0]
	err_array1 = [0]
	normalization0 = h_err_list[0].Integral(1,bins+1,1,bins+1)
	normalization1 = h_err_list[1].Integral(1,bins+1,1,bins+1)	    
	for i in range(1,bins+1):
	    err_array0.append(h_err_list[0].Integral(1,i,1,i)/normalization0)
	    err_array1.append(h_err_list[1].Integral(1,i,1,i)/normalization1)
	plt.plot(cut_array,err_array0, label="old", marker = "^" )
	plt.plot(cut_array,err_array1, label = "new", marker = "v")
	plt.legend()
	plt.xlabel("Bs2derr cut (cm)")
	plt.ylabel("Efficiency")
	plt.title("# Events vs Bs2derr cut")
	plt.savefig(folder[-1]+"/"+FILE+"/err_eff_ovn.png")
	plt.clf()
if nsvcel:
    c0.cd()
    nv_list[0].Draw()
    nv_list[1].Draw('Same')
    ROOT.gStyle.SetOptFit(0000)
    c0.BuildLegend()
    c0.Print(folder[-1]+"/"+FILE+"/nv_ovn.png")
if eff_dvv3D:
    c_dvv3D.cd()
    for i in range(len(dvv3D_list)):
	dvv3D_list[i].Draw('Same')
    c_dvv3D.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dvv3D.Print(folder[-1]+"/"+FILE+"/dVV3D_eff_ovn.png")
if eff_dvv:
    c_dvv.cd()
    for i in range(len(dvv_list)):
	dvv_list[i].Draw('Same')
    c_dvv.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dvv.Print(folder[-1]+"/"+FILE+"/dVV_eff_ovn.png")
if eff_dbv:
    c_dbv.cd()
    for i in range(len(dbv_list)):
	dbv_list[i].Draw('Same')
    c_dbv.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dbv.Print(folder[-1]+"/"+FILE+"/dBV_eff_ovn.png")
if eff_dbv3D:
    c_dbv3D.cd()
    for i in range(len(dbv3D_list)):
	dbv3D_list[i].Draw('Same')
    c_dbv3D.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_dbv3D.Print(folder[-1]+"/"+FILE+"/dBV3D_eff_ovn.png")
if deltaphi:
    c_phi.cd()
    for i in range(len(dbv3D_list)):
	dphi_list[i].Draw('Same')
    c_phi.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_phi.Print(folder[-1]+"/"+FILE+"/dphi_eff_ovn.png")
if deltaphi3D:
    c_phi3D.cd()
    for i in range(len(dbv3D_list)):
	dphi3D_list[i].Draw('Same')
    c_phi3D.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c_phi3D.Print(folder[-1]+"/"+FILE+"/dphi3D_eff_ovn.png")
if recodVV:
    c0.Clear()
    c0.cd()
    recodVV_list[1].Draw()
    recodVV_list[0].Draw('Same')
    c0.BuildLegend()
    ROOT.gStyle.SetOptFit(0000)
    c0.Print(folder[-1]+"/"+FILE+"/recodVV.png")
print str(integral[0]) + ", " + str(integral[1])+ ', ratio is' + str(integral[1]/integral[0])
fA.Close()
