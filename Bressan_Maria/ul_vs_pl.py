ƒ#!/usr/bin/env python
import ROOT
import sys, os, argparse, math
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from pprint import pprint
from ROOT import TLorentzVector
parser = argparse.ArgumentParser(description = 'Specify tau (in um) and mass (in GeV)')
parser.add_argument('folder', type=str)
parser.add_argument('file', type=str)
parser.add_argument('tau', type=str)
parser.add_argument('mass', t.ype=str)
args = parser.parse_args()

#folder = args.folder #"MiniTreeDvvREUsOldVtxV27m"

tau = args.tau
m = args.mass
filename = args.file

# Path to input root file
i = ["/nfs/cms/mc1/reu2022/"+folder+"/"+filename +".root"]

folder_list = ["MiniTreeREUsNewVtxNewGenULV27m", "MiniTreeDvvREUsNewVtxV27m"]

if not os.path.isdir(folder):
    os.mkdir(folder)
if not os.path.isdir(folder+"/"+filename):
    os.mkdir(folder+"/"+filename)

# Which plots do you want?
plot_masses = False # recommended to use the 5-vertex condition for Mass
plot_lfdisp = False # Plots lab frame displacement, including a fit
plot_rfdisp = False # Plots rest frame displacement, including a fit and log of fit
plot_gb = True # Plots gamma*beta
plot_dvv3D = False # Plots true distance between vertices
plot_dvv = False # Plots transverse distance between vertices

xx = True # gen_pv_x - bsx
yy = True # gen_pv_y - bsy
gen_0 = True # Gen_x0 vs gen_y0

bins = 100

tauint = float(tau)/1000
mint = int(m)
end = tauint/2

cutoff = tauint/5 #cutoff for fit

def findall(f, s):
    l = []
    i = -1
    while True:
        i = s.find(f, i+1)
        if i == -1:
            return l
        l.append(s.find(f, i))

def pltfit(h, title, folder, filename, c0, tau, m, cutoff):
    h.Draw()
    fit = h.Fit("expo", "S", "Draw", 0, cutoff)
    A = fit.Parameter(0)
    lbd = fit.Parameter(1)
    chi2 = fit.Chi2()
    #text = ROOT.TText(0.2,0.5,"A="+ str(A)[0:3]+" c*tau="+str(-1/lbd)[0:5]+" chi2="+str(chi2))
    #text.SetNDC()
    #text.Draw()
    ROOT.gStyle.SetOptFit(1111)
    c0.Update()
    #c0.Print(title+"_"+tau+"_mm_"+ m+"_GeV"+".root")
    c0.Print(folder+"/"+filename+"/"+title+".png")
    print str(-1/lbd)
    c0.Clear()

print "begin"

#filename_i_list = options.positional #FIXME : may consider using a wild card to make multiple histograms at once from multiple samples (in *.root files)
for folder in folder_list:
    first = findall('/', i)[-1]+1
    last = findall('.',i)[-1]
    name = i[first:last]
    fA = ROOT.TFile.Open(i)
    '''if not fA.IsOpen():
         raise IOError('could not open input file %s' % fnA)'''
    t = fA.Get("mfvMiniTreePreSelEvtFilt/t") #FIXME : this example opens a tree of a specific event category of >=2vtx >=5trk events (vertices in all event categories have all quality cuts applied except bs2derr)
    if plot_masses:
        h_mass = ROOT.TH1F("MC Stop->dd "+tau+" mm "+m+" GeV, Mass","Stop Mass; Mass (GeV); # Events", bins, 0, 1.2*mint)
    if plot_lfdisp:
        h_lfdisp = ROOT.TH1F("MC Stop->dd "+tau+" mm "+m+" GeV, Lab Frame Disp","Displacement in Lab Frame; Distance (cm); # Events", bins, 0, tauint/2)
    if plot_rfdisp or plot_gb:
	h_rfdisp = ROOT.TH1F("MC Stop->dd "+tau+" mm "+m+" GeV, Rest Frame Disp","Displacement in Rest Frame; Distance (cm); # Events", bins, 0, tauint/2)
	rf1, rf2 = TLorentzVector(), TLorentzVector()
    if plot_gb:
	h_gb = ROOT.TH1F("MC Stop->dd "+tau+" mm "+m+" GeV, gamma*beta","gamma*beta", bins, 0, 4)
    if plot_dvv3D:
	h_dvv3D = ROOT.TH1F("MC Stop->dd "+tau+" mm "+m+" GeV, dVV3D","dVV3D; Displacement (cm); # Events", bins, 0, tauint)
    if plot_dvv:
	h_dvv = ROOT.TH1F("MC Stop->dd "+tau+" mm "+m+" GeV, dVV","dVV; Displacement (cm); # Events", bins, 0, tauint/2)
    if xx:
	h_xx = ROOT.TH1F("xx","gen_pv_x - bsx; cm; # Events", bins, -0.005, 0.005)
    if yy:
	h_yy = ROOT.TH1F("yy","gen_pv_y - bsy; cm; # Events", bins, -0.005, 0.005)
    if gen_0:
	h_gen_0 = ROOT.TH2F("Gen_x0 vs gen_y0","Gen_x0 vs Gen_y0; Gen_x0 (cm); Gen_y0 (cm)", bins,-end,end,bins,-end,end)
    count_entry = 0
    for entry in t:
        evt_weight = entry.weight
        evt_run = entry.run
        evt_gen_daughters = entry.gen_daughters
        gen0, gen1, gen2, gen3 = TLorentzVector(), TLorentzVector(), TLorentzVector(), TLorentzVector()
	lfdisp_1 = ((entry.gen_x[0]+entry.bsx+entry.bsdxdz*(entry.gen_z[0])-entry.gen_pv_x0)**2 + (entry.gen_y[0]+entry.bsy+entry.bsdydz*(entry.gen_z[0])-entry.gen_pv_y0)**2+(entry.gen_z[0]+entry.bsz-entry.gen_pv_z0)**2)**(0.5)
	lfdisp_2 = ((entry.gen_x[1]+entry.bsx+entry.bsdxdz*(entry.gen_z[1])-entry.gen_pv_x0)**2 + (entry.gen_y[1]+entry.bsy+entry.bsdydz*(entry.gen_z[1])-entry.gen_pv_y0)**2+(entry.gen_z[1]+entry.bsz-entry.gen_pv_z0)**2)**(0.5)
	rat1 = (((entry.gen_x[0]+entry.bsx-entry.gen_pv_x0)**2 + (entry.gen_y[0]+entry.bsy-entry.gen_pv_y0)**2)**(0.5))/lfdisp_1
	rat2 = (((entry.gen_x[1]+entry.bsx-entry.gen_pv_x0)**2 + (entry.gen_y[1]+entry.bsy-entry.gen_pv_y0)**2)**(0.5))/lfdisp_2
	dvv3D = (((entry.gen_x[0]+entry.bsx-entry.gen_pv_x0)-(entry.gen_x[1]+entry.bsx-entry.gen_pv_x0))**2 + ((entry.gen_y[0]+entry.bsy-entry.gen_pv_y0)-(entry.gen_y[1]+entry.bsy-entry.gen_pv_y0))**2 + ((entry.gen_z[0]+entry.bsz-entry.gen_pv_z0)-(entry.gen_z[1]+entry.bsz-entry.gen_pv_z0))**2)**(0.5)
	dvv = (((entry.gen_x[0]+entry.bsx-entry.gen_pv_x0)-(entry.gen_x[1]+entry.bsx-entry.gen_pv_x0))**2 + ((entry.gen_y[0]+entry.bsy-entry.gen_pv_y0)-(entry.gen_y[1]+entry.bsy-entry.gen_pv_y0))**2)**(0.5) 
	if plot_rfdisp or plot_gb:
	    rf1.SetPtEtaPhiM(entry.gen_lsp_pt[0],entry.gen_lsp_eta[0],entry.gen_lsp_phi[0],entry.gen_lsp_mass[0])
	    rf2.SetPtEtaPhiM(entry.gen_lsp_pt[1],entry.gen_lsp_eta[1],entry.gen_lsp_phi[1],entry.gen_lsp_mass[1])
	    rfdisp_1 = lfdisp_1/rf1.Gamma()/rf1.Beta()
	    rfdisp_2 = lfdisp_2/rf2.Gamma()/rf2.Beta()
	    gb1 = rf1.Gamma()*rf1.Beta()
	    gb2 = rf2.Gamma()*rf2.Beta()
	    lfdBV1 = ((entry.gen_x[0]+entry.bsx-entry.gen_pv_x0)**2 + (entry.gen_y[0]+entry.bsy-entry.gen_pv_y0)**2)**(0.5)
	    lfdBV2 = ((entry.gen_x[1]+entry.bsx-entry.gen_pv_x0)**2 + (entry.gen_y[1]+entry.bsy-entry.gen_pv_y0)**2)**(0.5)
	    rfdBV1 = lfdBV1/rf1.Gamma()/rf1.Beta()
	    rfdBV2 = lfdBV2/rf2.Gamma()/rf2.Beta()
	count_gen = 0
	if xx:
	    h_xx.Fill(entry.gen_pv_x0 - entry.bsx, evt_weight)
	if yy:
	    h_yy.Fill(entry.gen_pv_y0 - entry.bsy, evt_weight)
	if gen_0:
	    h_gen_0.Fill(entry.gen_x[0],entry.gen_y[0],evt_weight)
	    h_gen_0.Fill(entry.gen_x[1],entry.gen_y[1],evt_weight)
        if plot_masses:
            for gen in evt_gen_daughters:
                count_gen += 1
                if count_gen == 1:
                    gen0.SetPxPyPzE(gen.Px(), gen.Py(), gen.Pz(), gen.Energy())
                if count_gen == 2:
                    gen1.SetPxPyPzE(gen.Px(), gen.Py(), gen.Pz(), gen.Energy())
                if count_gen == 3:
                    gen2.SetPxPyPzE(gen.Px(), gen.Py(), gen.Pz(), gen.Energy())
                if count_gen == 4:
                    gen3.SetPxPyPzE(gen.Px(), gen.Py(), gen.Pz(), gen.Energy())
        if evt_run == 1:
            count_entry += 1
            if plot_masses:
                ep_vec_1 = gen0 + gen1 
	        ep_vec_2 = gen2 + gen3
                evt_mass_1 = ep_vec_1.Mag()
	        evt_mass_2 = ep_vec_2.Mag()
		h_mass.Fill(evt_mass_1, evt_weight)
		h_mass.Fill(evt_mass_2, evt_weight)
            if plot_lfdisp:
		h_lfdisp.Fill(lfdisp_1, evt_weight)
		h_lfdisp.Fill(lfdisp_2, evt_weight)
	    if plot_rfdisp:
		h_rfdisp.Fill(rfdisp_1, evt_weight)
		h_rfdisp.Fill(rfdisp_2, evt_weight)
	    if plot_gb:
		h_gb.Fill(gb1, evt_weight)
		h_gb.Fill(gb2, evt_weight)
	    if plot_dvv3D:
		h_dvv3D.Fill(dvv3D,evt_weight)
	    if plot_dvv:
		h_dvv.Fill(dvv,evt_weight)
    c0 = ROOT.TCanvas()
    if plot_masses:
        h_mass.Draw()
        c0.Update()
        #c0.Print("stop_mass_"+tau+"_mm_"+ m+"_GeV"+".root")
        c0.Print(folder+"/tau"+tau+"000um_M0"+m+"GeV/mass.png")
	print folder+"/tau"+tau+"000um_M0"+m+"GeV/mass.png"
        c0.Clear()
    if plot_lfdisp:
	pltfit(h_lfdisp, "lfdisp",folder, filename, c0, tau, m, cutoff)
    if plot_gb:
	h_gb.Draw()
        #c0.Print("gb_"+tau+"_mm_"+ m+"_GeV"+".root")
	c0.Print(folder+"/tau"+tau+"000um_M0"+m+"GeV/gb.png")
	c0.Clear()
    if plot_dvv3D:
	h_dvv3D.Draw()
	#c0.Print("dVV3D_"+tau+"_mm_"+ m+"_GeV"+".root")
	c0.Print(folder+"/tau"+tau+"000um_M0"+m+"GeV/dVV3D.png")
    if plot_dvv:
	h_dvv.Draw()
	#c0.Print("dVV_"+tau+"_mm_"+ m+"_GeV"+".root")
	c0.Print(folder+"/tau"+tau+"000um_M0"+m+"GeV/dVV.png")
	c0.Clear()
    if plot_rfdisp:
	'''log = []
	xinit = np.linspace(0,cutoff,bins)
	x = []
	for i in range(bins):
	    if h_rfdisp.GetBinContent(i)!=0:
		log.append(math.log(h_rfdisp.GetBinContent(i)))
		x.append((xinit[i]+xinit[i+1])/2)
	slope, b = np.polyfit(x, log, 1)
	plt.plot(xinit, slope*xinit+b, label="y="+str(round(slope,4))+"x+"+str(round(b,4)))
	plt.plot(x, log, "o")
	plt.title("Rest Frame Displacement (Log)")
	plt.ylabel("Log of # Events")
	plt.xlabel("Displacement (cm)")
	leg = plt.legend(loc='upper center')
	plt.savefig("log_rfdisp_"+tau+"_um_"+ m+"_GeV"+".png")	
	plt.clf()'''	
	pltfit(h_rfdisp, "rfdisp",folder, filename, c0, tau, m, cutoff)
        h_rfdisp.Fit("expo", "S0", "0", 0, cutoff)
        ROOT.gStyle.SetOptFit(1111)
	h_rfdisp.SetMinimum(0.1)
	h_rfdisp.Draw()
	c0.SetLogy(1)
	expfit = h_rfdisp.GetFunction("expo")
	expfit.SetRange(0,10)
	expfit.Draw("same")
	#c0.Print("rfdisp_logh_"+tau+"_mm_"+ m+"_GeV"+".root")
	c0.Print(folder+"/"+filename + "/rfdisp_log.png")
	c0.Clear()
    if xx:
	h_xx.Draw()
	c0.Print(folder+"/"+filename + "/xx.png")
	c0.Clear()
    if yy:
	h_yy.Draw()
	c0.Print(folder+"/"+filename + "/yy.png")
	c0.Clear()
    if gen_0:
	h_gen_0.Draw('colz')
	c0.Print(folder+"/"+filename + "/gen_0.png")
	c0.Clear()
    fA.Close()
