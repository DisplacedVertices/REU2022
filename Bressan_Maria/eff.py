#!/usr/bin/env python
import ROOT
import sys, os, argparse, math
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pprint import pprint
from ROOT import TLorentzVector
parser = argparse.ArgumentParser(description = 'folder, file (without .root)')
parser.add_argument('folder', type=str)
parser.add_argument('file', type=str)
parser.add_argument('tau', type=str) #in um
parser.add_argument('mass', type=str)
args = parser.parse_args()


# Which plots do you want?
eff_dvv3D = False # Plots efficiency vs dvv3D
deltaphi3D = False # Graphs efficiency vs delta phi3D AND the two delta phi3D graphs overlayed

eff_dbv = False # Plots efficiency vs dbv
eff_dbv3D = False # Plots efficiency vs dbv3D

eff_dvv = False  # Plots efficiency vs dvv
deltaphi = False # Graphs efficiency vs delta phi AND the two delta phi graphs overlayed

gvr_dvv = False # Generated vs Reconstructed, dvv. All run with file s (post)
gvr_dbv = False 
gvr_dbv3D = False
gvr_dvv3D = False # Generated vs Reconstructed, dvv3D
gvr_phi = False # Generated vs Reconstructed, phi
gvr_phi3D = False # Generated vs Reconstructed, phi3D

err = True #Plots bs2derr of vertices
eff_err = True # Plots num events as a function of bs2derr cut

recogenratdBV = False # Ratio of reco dBV to gen dBV3D. After cuts and must have at least two tracks per event
recoratdBV = False # Ratio of reco dBV to reco dBV3D. After cuts and must have at least two tracks per event

recogenratdVV = False # Plot ratio of reconstructed transverse displacement to gen-level 3D displacement
recoratdVV = False # Plots ratio of reco transverse to reco 3D displacement

minphi = True # Plots minimum phi vs 2D distance
plot_nv = False # Plot number of vertecies

pre = "mfvMiniTreePreSelEvtFilt/t" # before cuts
post = "mfvMiniTree/t" # after cuts

bins = 100

folder = args.folder
FILE = args.file
bs2derr_cut = 0.0025
tau = args.tau
m = args.mass
M=m

filename_i_list = ["/nfs/cms/mc1/reu2022/"+folder+"/"+FILE+".root"]

if not os.path.isdir(folder):
    os.mkdir(folder)
if not os.path.isdir(folder+"/"+FILE):
    os.mkdir(folder+"/"+FILE)

print "begin " +folder+"/"+FILE

tauint = float(tau)/1000
mint = int(m)
end3D = tauint # Graphs x-range in cm: (0, end) for dvv3D (usually 10 if tau=10mm, 5 if tau=1mm)
end = tauint/2 # Graphs x-range in cm: (0, end) for dvv (usually 5 if tau=10mm, 2.5 if tau=1mm)

def nv(filename, tree, savefile):
    c0 = ROOT.TCanvas()
    fA = ROOT.TFile.Open(filename)
    t = fA.Get(tree)
    h = ROOT.TH1F("Number of Vertices","# Vertices after Cuts: " +post[0:-2] +"; # Vertices; # Events", 5, -0.5, 4.5)
    for entry in t:
	count=0
	for v in entry.vertices:
	    count+=1
	h.Fill(count, entry.weight)
    h.Draw()
    c0.Print(savefile)
    c0.Clear()

for i in filename_i_list:
    fA = ROOT.TFile.Open(i)
    t = fA.Get(pre)
    s = fA.Get(post)
    if plot_nv:
	nv_list = nv(i, post, folder+"/"+FILE+"/nv.png")
    if eff_dvv3D:
	h_dvv3D = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV3D truth","Gen-Level dVV3D Efficiency; Displacement (cm); Efficiency", bins, 0, end3D)
	h_dvv3D_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV3D","dVV3D Efficiency; Displacement (cm); Efficiency", bins, 0, end3D)
    if eff_dvv:
	h_dvv = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV truth","Gen-Level dVV Efficiency; Displacement (cm); Efficiency", bins, 0, end)
	h_dvv_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dVV","dVV Efficiency; Displacement (cm); Efficiency", bins, 0, end)
    if eff_dbv:
	h_dbv = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV truth","Gen-Level dBV Efficiency; Displacement (cm); Efficiency", bins, 0, end)
	h_dbv_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV","Gen-Level dBV Efficiency; Displacement (cm); Efficiency", bins, 0, end)
    if eff_dbv3D:
	h_dbv3D = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV3D truth","Gen-Level dBV3D Efficiency; Displacement (cm); Efficiency", bins, 0, end3D)
	h_dbv3D_sel = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, dBV3D","Gen-Level dBV3D Efficiency; Displacement (cm); Efficiency", bins, 0, end3D)
    if deltaphi3D or deltaphi:
	h_dp3D_t = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi 3D truth","Gen-Level delta Phi 3D; delta Phi; # Events", bins, 0, 3.15)
	h_dp3D_s = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi 3D selected","Efficiency vs Delta Phi 3D; delta Phi; Efficiency", bins, 0, 3.15)
	h_dp_t = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi truth","Gen-Level delta Phi; delta Phi; # Events", bins, 0, 3.15)
	h_dp_s = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, delta Phi selected","Efficiency vs Delta Phi; delta Phi; Efficiency", bins, 0, 3.15)
    if recogenratdVV:
	h_recogenratdVV = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, Reco dVV/Gen dVV3D","Ratio of Reconstructed dVV to gen dVV3D; ratio; # Events", bins, 0, 1.2)
    if recoratdVV:
	h_recoratdVV = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, Reco dVV/dVV3D","Ratio of Reconstructed dVV to dVV3D; ratio; # Events", bins, 0, 1)
    if recoratdBV:
	h_recoratdBV = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, Reco dBV/dBV3D","Ratio of Reconstructed dBV to dBV3D; ratio; # Events", bins, 0, 1)
    if recogenratdBV:
	h_recogenratdBV = ROOT.TH1F("MC Stop->dd "+tau+" mm "+M+" GeV, Reco dBV/Gen dBV3D","Ratio of Reconstructed dBV to Gen dBV3D; ratio; # Events", bins, 0, 1.2)
    if gvr_dvv or gvr_dvv3D or gvr_phi or gvr_phi3D:
	h_gvr_dvv = ROOT.TH2F("dvv","Gen vs Reco dvv; Gen dvv (cm); Reconstructed dvv (cm)", bins,0,end,bins,0,end)
	h_gvr_dvv3D = ROOT.TH2F("dvv3D","Gen vs Reco dvv3D; Gen dvv3D (cm); Reconstructed dvv3D (cm)", bins,0,end3D,bins,0,end3D)
	h_gvr_phi = ROOT.TH2F("phi","Gen vs Reco phi; Gen phi (rad); Reconstructed phi (rad)", bins,0,3.15,bins,0,3.15)	
	h_gvr_phi3D = ROOT.TH2F("phi3D","Gen vs Reco phi3D; Gen phi3D (rad); Reconstructed phi3D (rad)", bins,0,3.15,bins,0,3.15)
    if gvr_dbv:
	h_gvr_dbv = ROOT.TH2F("dbv","Gen vs Reco dbv; Gen dbv (cm); Reconstructed dbv (cm)", bins,0,end,bins,0,end)
    if gvr_dbv3D:
	h_gvr_dbv3D = ROOT.TH2F("dbv3D","Gen vs Reco dbv3D; Gen dbv3D (cm); Reconstructed dbv3D (cm)", bins,0,end3D,bins,0,end3D)
    if minphi:
	h_minphi = ROOT.TH2F("minphi","Minimum dPhi vs 2D Distance; Minimum dPhi; 2D distance (cm)", bins,0,1,bins,0,end/4)
    if err:
	h_err = ROOT.TH2F("bs2derr","Bs2derr; bs2derr of 1st vertex (cm); bs2derr of 2nd vertex (cm)", bins,0,0.01,bins,0,0.01)
    nvertex_list = []
    count_entry = 0
    for entry in s:
	evt_weight = entry.weight
	evt_run = entry.run
	evt_gen_daughters = entry.gen_daughters
	bsx = entry.bsx
	bsy = entry.bsy
	x = [entry.gen_x[0]+entry.bsx-entry.gen_pv_x0, entry.gen_x[1]+entry.bsx-entry.gen_pv_x0]
	y = [entry.gen_y[0]+entry.bsy-entry.gen_pv_y0, entry.gen_y[1]+entry.bsy-entry.gen_pv_y0]
	z = [entry.gen_z[0]+entry.bsz-entry.gen_pv_z0, entry.gen_z[1]+entry.bsz-entry.gen_pv_z0]
	lfdisp_1 = (x[0]**2 + y[0]**2+z[0]**2)**(0.5)
	lfdisp_2 = (x[1]**2 + y[1]**2+z[1]**2)**(0.5)
	dvv3D = ((x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2)**(0.5)
	dvv = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**(0.5) 
	dphi = abs(entry.gen_lsp_phi[0]-entry.gen_lsp_phi[1])
	dot = (x[0]*x[1]+y[0]*y[1]+z[0]*z[1])/lfdisp_1/lfdisp_2
	if dot > 0.9999:
	    dphi3D = 0
	else:
	    dphi3D = math.acos(dot)
	if True:
	    count = 0
	    xr = []
	    yr = []
	    zr = []
	    phi = []
	    for v in entry.vertices:
		if v.rescale_bs2derr < bs2derr_cut: 
		    xr.append(v.x) #reconstructed x,y,z coordinates
		    yr.append(v.y)
		    zr.append(v.z)
		    phi.append(math.atan((v.y-bsy)/(v.x-bsx)))
	    if len(xr)>1:
		if gvr_dvv:
	            h_gvr_dvv.Fill(dvv,((xr[0]-xr[1])**2+(yr[0]-yr[1])**2)**(0.5),evt_weight) #generated (x) vs reconstructed (y)
		if gvr_dvv3D:
	            h_gvr_dvv3D.Fill(dvv3D,((xr[0]-xr[1])**2+(yr[0]-yr[1])**2+(zr[0]-zr[1])**2)**(0.5),evt_weight)
		dotr = (xr[0]*xr[1]+yr[0]*yr[1])/(xr[0]**2+yr[0]**2)**(0.5)/(xr[1]**2+yr[1]**2)**(0.5)
		if dotr > 0.9999:
		    rphi=0
		else:
		    rphi = math.acos(dotr)
		if gvr_phi:
		        h_gvr_phi.Fill(dphi,rphi,evt_weight)
		dotr3D = (xr[0]*xr[1]+yr[0]*yr[1]+zr[0]*zr[1])/(xr[0]**2+yr[0]**2+zr[0]**2)**(0.5)/(xr[1]**2+yr[1]**2+zr[1]**2)**(0.5)
		if dotr3D > 0.9999:
		    rphi3D = 0
		else:
		    rphi3D = math.acos(dotr3D)
		dotg3D = (entry.gen_x[0]*entry.gen_x[1]+entry.gen_y[0]*entry.gen_y[1]+entry.gen_z[0]*entry.gen_z[1])/(entry.gen_x[0]**2+entry.gen_y[0]**2+entry.gen_z[0]**2)**(0.5)/(entry.gen_x[1]**2+entry.gen_y[1]**2+entry.gen_z[1]**2)**(0.5)
		if dotg3D > 0.9999:
		    gphi3D = 0
		else:
		    gphi3D = math.acos(dotg3D)
		if gvr_phi3D: 
		    h_gvr_phi3D.Fill(gphi3D,rphi3D,evt_weight)
		if gvr_dbv:
		    dbv1 = (entry.gen_x[0]**2 + entry.gen_y[0]**2)**(0.5)
		    dbv2 = (entry.gen_x[1]**2 + entry.gen_y[1]**2)**(0.5)
		    rdbv1 = (xr[0]**2 + yr[0]**2)**(0.5)
		    rdbv2 = (xr[1]**2 + yr[1]**2)**(0.5)
		    if abs(dbv1-rdbv1) < abs(dbv1-rdbv2):
		        h_gvr_dbv.Fill(dbv1,rdbv1,evt_weight)
		        h_gvr_dbv.Fill(dbv2,rdbv2,evt_weight)
		    else:
		        h_gvr_dbv.Fill(dbv2,rdbv1,evt_weight)
		        h_gvr_dbv.Fill(dbv1,rdbv2,evt_weight)			
		if gvr_dbv3D:
		    dbv3D1 = (entry.gen_x[0]**2 + entry.gen_y[0]**2 + entry.gen_z[0]**2)**(0.5)
		    dbv3D2 = (entry.gen_x[1]**2 + entry.gen_y[1]**2 + entry.gen_z[1]**2)**(0.5)
		    rdbv3D1 = (xr[0]**2 + yr[0]**2 + zr[0]**2)**(0.5)
		    rdbv3D2 = (xr[1]**2 + yr[1]**2 + zr[1]**2)**(0.5)
		    if abs(dbv3D1-rdbv3D1) < abs(dbv3D1-rdbv3D2):
		        h_gvr_dbv3D.Fill(dbv3D1,rdbv3D1,evt_weight)
		        h_gvr_dbv3D.Fill(dbv3D2,rdbv3D2,evt_weight)
		    else:
		        h_gvr_dbv3D.Fill(dbv3D2,rdbv3D1,evt_weight)
		        h_gvr_dbv3D.Fill(dbv3D1,rdbv3D2,evt_weight)	
		if True:
		    if recogenratdVV:
		        h_recogenratdVV.Fill(((xr[0]-xr[1])**2+(yr[0]-yr[1])**2)**(0.5)/dvv3D,evt_weight)
		    if recoratdVV:
		        h_recoratdVV.Fill(((xr[0]-xr[1])**2+(yr[0]-yr[1])**2)**(0.5)/((xr[0]-xr[1])**2+(yr[0]-yr[1])**2+(zr[0]-zr[1])**2)**(0.5),evt_weight)
		    if recoratdBV:
		        h_recoratdBV.Fill((xr[0]**2+yr[0]**2)**(0.5)/(xr[0]**2+yr[0]**2+zr[0]**2)**(0.5),evt_weight)
		        h_recoratdBV.Fill((xr[1]**2+yr[1]**2)**(0.5)/(xr[1]**2+yr[1]**2+zr[1]**2)**(0.5),evt_weight)
	    	    if recogenratdBV:
		        h_recogenratdBV.Fill((xr[0]**2+yr[0]**2)**(0.5)/lfdisp_1,evt_weight)
		        h_recogenratdBV.Fill((xr[1]**2+yr[1]**2)**(0.5)/lfdisp_2,evt_weight)
	    	    if eff_dvv3D:
			h_dvv3D_sel.Fill(((xr[0]-xr[1])**2+(yr[0]-yr[1])**2+(zr[0]-zr[1])**2)**(0.5),evt_weight)
	    	    if eff_dvv:
			h_dvv_sel.Fill(((xr[0]-xr[1])**2+(yr[0]-yr[1])**2)**(0.5),evt_weight)
	    	    if deltaphi:
			h_dp_s.Fill(rphi, evt_weight)
	    	    if deltaphi3D:
			h_dp3D_s.Fill(rphi3D, evt_weight)
	    	    if eff_dbv:
			h_dbv_sel.Fill((xr[0]**2+yr[0]**2)**(0.5), evt_weight)
			h_dbv_sel.Fill((xr[1]**2+yr[1]**2)**(0.5), evt_weight)
	    	    if eff_dbv3D:
			h_dbv3D_sel.Fill((xr[0]**2+yr[0]**2+zr[0]**2)**(0.5),evt_weight)
			h_dbv3D_sel.Fill((xr[1]**2+yr[1]**2+zr[1]**2)**(0.5),evt_weight)
		if len(xr)==3:
		    if minphi:
			dphivals = [abs(phi[0]-phi[1]),abs(phi[0]-phi[2]),abs(phi[1]-phi[2])] # 12, 13, 23
			dVVvals = [((xr[0]-xr[1])**2+(yr[0]-yr[1])**2)**(0.5),((xr[0]-xr[2])**2+(yr[0]-yr[2])**2)**(0.5),((xr[2]-xr[1])**2+(yr[2]-yr[1])**2)**(0.5)]
			minphival = np.min(dphivals)
			mindVVval = dVVvals[dphivals.index(minphival)]
			h_minphi.Fill(minphival,mindVVval,evt_weight)
    for entry in t:
        evt_weight = entry.weight
        evt_run = entry.run
        evt_gen_daughters = entry.gen_daughters
	if eff_dvv3D or eff_dvv or deltaphi or deltaphi3D or eff_dbv or eff_dbv3D:
	    x = [entry.gen_x[0]+entry.bsx-entry.gen_pv_x0, entry.gen_x[1]+entry.bsx-entry.gen_pv_x0]
	    y = [entry.gen_y[0]+entry.bsy-entry.gen_pv_y0, entry.gen_y[1]+entry.bsy-entry.gen_pv_y0]
	    z = [entry.gen_z[0]+entry.bsz-entry.gen_pv_z0, entry.gen_z[1]+entry.bsz-entry.gen_pv_z0]
	    lfdisp_1 = (x[0]**2 + y[0]**2+z[0]**2)**(0.5)
	    lfdisp_2 = (x[1]**2 + y[1]**2+z[1]**2)**(0.5)
	    dvv3D = ((x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2)**(0.5)
	    dvv = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**(0.5) 
	    dphi = abs(entry.gen_lsp_phi[0]-entry.gen_lsp_phi[1])
	    dot = (x[0]*x[1]+y[0]*y[1]+z[0]*z[1])/lfdisp_1/lfdisp_2
	    if dot > 0.9999:
		dphi3D = 0
	    else:
		dphi3D = math.acos(dot)
	count_gen = 0
        if evt_run == 1:
            count_entry += 1
	    if err:
		err_list = []
		for v in entry.vertices:
		    err_list.append(v.bs2derr)
		if len(err_list)>1:
		    h_err.Fill(err_list[0],err_list[1],evt_weight)
	    if eff_dvv3D:
		h_dvv3D.Fill(dvv3D,evt_weight)
	    if eff_dvv:
		h_dvv.Fill(dvv,evt_weight)
	    if deltaphi:
		h_dp_t.Fill(dphi, evt_weight)
	    if deltaphi3D:
		h_dp3D_t.Fill(dphi3D, evt_weight)
	    if eff_dbv:
		h_dbv.Fill((x[0]**2+y[0]**2)**(0.5),evt_weight)
		h_dbv.Fill((x[1]**2+y[1]**2)**(0.5),evt_weight)
	    if eff_dbv3D:
		h_dbv3D.Fill((x[0]**2+y[0]**2+z[0]**2)**(0.5),evt_weight)
		h_dbv3D.Fill((x[1]**2+y[1]**2+z[1]**2)**(0.5),evt_weight)
    c0 = ROOT.TCanvas()
    if eff_err:
	cut_array = np.linspace(0,0.01,bins+1)
	err_array = [0]
	normalization = h_err.Integral(1,bins+1,1,bins+1)
	for i in range(1,bins+1):
	    err_array.append(h_err.Integral(1,i,1,i)/normalization)
	plt.plot(cut_array,err_array)
	plt.xlabel("Bs2derr cut (cm)")
	plt.ylabel("Efficiency")
	plt.title("# Events vs Bs2derr cut")
	plt.savefig(folder+"/"+FILE+"/eff_err.png")
	plt.clf()
    if eff_dvv3D:
	hratio = h_dvv3D_sel.Clone("Efficiencies for dvv3D")
	hratio.Divide(hratio, h_dvv3D, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dVV3D_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dVV3D_eff_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if eff_dvv:
	hratio = h_dvv_sel.Clone("Efficiencies for dvv")
	hratio.Divide(hratio, h_dvv, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dVV_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dVV_eff_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if eff_dbv:
	hratio = h_dbv_sel.Clone("Efficiencies for dbv")
	hratio.Divide(hratio, h_dbv, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dVV_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dBV_eff_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if eff_dbv3D:
	hratio = h_dbv3D_sel.Clone("Efficiencies for dbv3D")
	hratio.Divide(hratio, h_dbv3D, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dVV_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dBV3D_eff_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if deltaphi:
	h_dp_t.Draw()
	h_dp_s.SetLineColor(6)
	h_dp_s.Draw("Same")
	c0.BuildLegend()
	c0.Print(folder+"/"+FILE+"/dphi_compare_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
	hratio = h_dp_s.Clone("Efficiencies for delta Phi")
	hratio.Divide(hratio, h_dp_t, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dp_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dphi_eff_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if deltaphi3D:
	h_dp3D_t.Draw()
	h_dp3D_s.SetLineColor(6)
	h_dp3D_s.Draw("Same")
	c0.BuildLegend()
	c0.Print(folder+"/"+FILE+"/dphi3D_compare_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
	hratio = h_dp3D_s.Clone("Efficiencies for delta Phi 3D")
	hratio.Divide(hratio, h_dp3D_t, 1, 1, "B")
	hratio.Draw()
	#c0.Print("dp3D_eff_"+tau+"_mm_"+M+"_GeV"+".root")
	c0.Print(folder+"/"+FILE+"/dphi3D_eff_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if gvr_dvv:	
	h_gvr_dvv.Draw("colz")
	c0.Print(folder+"/"+FILE+"/gvr_dvv_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
	integral = 0
	for x in range(10,150):
	    integral += h_gvr_dvv.Integral(x,bins,x-10,x-10)
	for x in range(0,bins-10):
	    integral += h_gvr_dvv.Integral(0,x,x+10,bins)
	print integral
    if gvr_dvv3D:
	h_gvr_dvv3D.Draw("colz")
	c0.Print(folder+"/"+FILE+"/gvr_dvv3D_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if gvr_dbv:
	h_gvr_dbv.Draw("colz")
	c0.Print(folder+"/"+FILE+"/gvr_dbv.png")
	c0.Clear()
    if gvr_dbv3D:
	h_gvr_dbv3D.Draw("colz")
	c0.Print(folder+"/"+FILE+"/gvr_dbv3D.png")
	c0.Clear()
    if gvr_phi:
	h_gvr_phi.Draw("colz")
	c0.Print(folder+"/"+FILE+"/gvr_dphi_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if gvr_phi3D:
	h_gvr_phi3D.Draw("colz")
	c0.Print(folder+"/"+FILE+"/gvr_dphi3D_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if recogenratdVV:
	h_recogenratdVV.Draw()
	c0.Print(folder+"/"+FILE+"/recogenratdVV_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if recoratdVV:
	h_recoratdVV.Draw()
	c0.Print(folder+"/"+FILE+"/recoratdVV_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if recoratdBV:
	h_recoratdBV.Draw()
	c0.Print(folder+"/"+FILE+"/recoratdBV_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if recogenratdBV:
	h_recogenratdBV.Draw()
	c0.Print(folder+"/"+FILE+"/recogenratdBV_bs2derr"+str(bs2derr_cut)+".png")
	c0.Clear()
    if minphi:
	h_minphi.Draw('colz')
	c0.Print(folder+"/"+FILE+"/minphi.png")
	c0.Clear()
    if err:
	h_err.Draw('colz0')
	c0.Print(folder+"/"+FILE+"/err.png")
	c0.Clear()
    fA.Close()
