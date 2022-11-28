#!/usr/bin/env python

"""Creates 10mm reweight dBV plots."""

import ROOT
import sys, os, argparse
import math
from glob import glob
from ROOT import TLorentzVector

parser = argparse.ArgumentParser(
    description='run an analysis on *.root minitree files in a given pathA',
    usage='%(prog)s [options] pathA')
parser.add_argument('positional', nargs='*')
options = parser.parse_args()

pathA = options.positional[0]
filename_i_list = glob(pathA + 'ggHToSSTodddd_tau10mm_M55_2017.root') #FIXME : subject to change file name and vary mass-point samples

prestudy = False # FIXME : turn on before knowing the size of a sample to be requested  
tot_evt_to_req = 500000 # FIXME : a fixed size of a desire sample request
dp = 0.7 # FIXME : estimate efficiency drop
end = 10 
bins = 50
lfbins = 50
weight_on = 1 # FIXME : the reweighting starts with lumi*xsec weights off but turns it on to get the final signal yieds with a desire size of samples 
if not weight_on :
    print("** Rewighting procedure will be done with event weight(lumi*xsec) OFF **")
    evt_weight = 1.0
else :
    print("** Rewighting procedure will be done with event weight(lumi*xsec) ON **") 
limit = False

ROOT.TH1.SetDefaultSumw2()
print("Begin...")
for i in filename_i_list:
    c0 = ROOT.TCanvas()
    i_list = i.split("_") #FIXME
    tau = str(i_list[1]) #FIXME
    tau = tau[3:] #FIXME
    m = str(i_list[2]) #FIXME
    m = m[1:] #FIXME
    fA = ROOT.TFile.Open(i)
    j = i.replace("MiniTree","HistosFinalized") #FIXME
    fB = ROOT.TFile.Open(j) 
    if not fA.IsOpen():
        raise IOError('could not open input file %s' % fnA)
    t = fA.Get("mfvMiniTree/t")
    s = fA.Get("mfvMiniTreePreSelEvtFilt/t")
    h = fB.Get('mfvEventHistosNoCuts').Get('h_jet_gen_ht_30') #FIXME : path to a histogram filled by total genrated events  
    h_sums = fB.Get('mfvWeight').Get('h_sums')
    h_rfdisp_1 = ROOT.TH1F(  # Rest Frame dBV LLP1
        "MC Higgs->ss " + tau + " " + m + " GeV",
        "H->LLP1->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP1 Reconstructed Vertex and "
                                           "Beamspot (cm); # Events", 50, 0, end)
    h_rfdisp_1_scale_30mm = ROOT.TH1F(  # Rest Frame dBV LLP1 ->30mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 30mm",
        "Displacement in Rest Frame (1); 2D Separation of Reconstructed Vertex and "
        "Beamspot (cm); # Events", 50, 0, end)
    h_rfdisp_1_scale_3mm = ROOT.TH1F(  # Rest Frame dBV LLP1 ->3mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 3mm",
        "Displacement in Rest Frame (1); 2D Separation of Reconstructed Vertex and "
        "Beamspot (cm); # Events", 50, 0, end)
    h_rfdisp_2 = ROOT.TH1F(  # Rest Frame dBV LLP2
        "MC Higgs->ss " + tau + " " + m + " GeV, ",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP2 Reconstructed Vertex and "
                                           "Beamspot (cm); # Events", 50, 0, end)
    h_rfdisp_2_scale_30mm = ROOT.TH1F(  # Rest Frame dBV LLP2 ->30mm
        "MC Higgs->ss " + tau + " " + m + " GeV,  Scale -> 30mm",
        "Displacement in Rest Frame (2); 2D Separation of Reconstructed Vertex and "
        "Beamspot (cm); # Events", 50, 0, end)
    h_rfdisp_2_scale_3mm = ROOT.TH1F(  # Rest Frame dBV LLP2 ->3mm
        "MC Higgs->ss " + tau + " " + m + " GeV,  Scale -> 3mm",
        "Displacement in Rest Frame (2); 2D Separation of Reconstructed Vertex and "
        "Beamspot (cm); # Events", 50, 0, end)
    h_cut_gen_jet_ht30 = ROOT.TH1F(
        "h_cut_gen_jet_ht_30", 
        "with selection cuts;"
        "gen H_{T} of jets with p_{T} > 30 GeV;"
        "events/25 GeV", 1000, 0, 5000) 
    h_30mm_cut_gen_jet_ht30 = ROOT.TH1F(
        "h_30mm_cut_gen_jet_ht_30", 
        "with selection cuts;"
        "gen H_{T} of jets with p_{T} > 30 GeV;"
        "events/25 GeV", 1000, 0, 5000) 
    h_3mm_cut_gen_jet_ht30 = ROOT.TH1F(
        "h_3mm_cut_gen_jet_ht_30", 
        "with selection cuts;"
        "gen H_{T} of jets with p_{T} > 30 GeV;"
        "events/25 GeV", 1000, 0, 5000) 
    rf1, rf2 = TLorentzVector(), TLorentzVector()

    count_s_entry = 0
    for entry in s:  #loop over all events to contruct weight plots 
        count_s_entry += 1
        if limit and count_s_entry > 500:
            print("ENTRY LIMIT IS ON!")
            break
        evt_weight = 1.0 # to disentangle lumi*xsec weights in the procedure 
        evt_run = entry.run
        evt_bs2derr_0 = entry.bs2derr0
        evt_bs2derr_1 = entry.bs2derr1
        evt_gen_daughters = entry.gen_daughters
        evt_gen_jet_ht30 = entry.gen_jet_ht30
        gen0, gen1, gen2, gen3 = TLorentzVector(), TLorentzVector(), TLorentzVector(), TLorentzVector()
        lfdisp_1 = ((entry.bsx + entry.gen_x[0] - entry.gen_pv_x0) ** 2 + (
                entry.bsy + entry.gen_y[0] - entry.gen_pv_y0) ** 2 + (
                            entry.bsz + entry.gen_z[0] - entry.gen_pv_z0) ** 2) ** 0.5
        lfdisp_2 = ((entry.bsx + entry.gen_x[1] - entry.gen_pv_x0) ** 2 + (
                entry.bsy + entry.gen_y[1] - entry.gen_pv_y0) ** 2 + (
                            entry.bsz + entry.gen_z[1] - entry.gen_pv_z0) ** 2) ** 0.5
        rf1.SetPtEtaPhiM(entry.gen_lsp_pt[0], entry.gen_lsp_eta[0], entry.gen_lsp_phi[0], entry.gen_lsp_mass[0])
        rf2.SetPtEtaPhiM(entry.gen_lsp_pt[1], entry.gen_lsp_eta[1], entry.gen_lsp_phi[1], entry.gen_lsp_mass[1])
        rfdisp_1 = lfdisp_1 / rf1.Gamma() / rf1.Beta()
        rfdisp_2 = lfdisp_2 / rf2.Gamma() / rf2.Beta()

        h_rfdisp_1.Fill(rfdisp_1, evt_weight)
        h_rfdisp_2.Fill(rfdisp_2, evt_weight)
        h_rfdisp_1_scale_30mm.Fill(rfdisp_1, evt_weight)
        h_rfdisp_2_scale_30mm.Fill(rfdisp_2, evt_weight)
        h_rfdisp_1_scale_3mm.Fill(rfdisp_1, evt_weight)
        h_rfdisp_2_scale_3mm.Fill(rfdisp_2, evt_weight)
    rfit1 = ROOT.TF1("rfit1", "expo", 0, 10)
    rfit1.SetLineColor(ROOT.kBlue)
    h_rfdisp_1.Fit("rfit1")
    rfit2 = ROOT.TF1("rfit2", "expo", 0, 10)
    rfit2.SetLineColor(ROOT.kRed)
    h_rfdisp_1_scale_30mm.Fit("rfit2")
    rfit3 = ROOT.TF1("rfit3", "expo", 0, 10)
    rfit3.SetLineColor(ROOT.kMagenta)
    h_rfdisp_1_scale_3mm.Fit("rfit3")
    rfit4 = ROOT.TF1("rfit4", "expo", 0, 10)
    rfit4.SetLineColor(ROOT.kBlue)
    h_rfdisp_2.Fit("rfit4")
    rfit5 = ROOT.TF1("rfit5", "expo", 0, 10)
    rfit5.SetLineColor(ROOT.kRed)
    h_rfdisp_2_scale_30mm.Fit("rfit5")
    rfit6 = ROOT.TF1("rfit6", "expo", 10)
    rfit6.SetLineColor(ROOT.kMagenta)
    h_rfdisp_2_scale_3mm.Fit("rfit6")

    rfit1.SetParName(0, "constant1")
    rfit1.SetParName(1, "slope1")
    rfit1.SetParameter("slope1", -1.0 / 1.0)
    rfit2.SetParName(0, "constant2")
    rfit2.SetParName(1, "slope2")
    rfit2.SetParameter("slope2", -1.0 / 3.0)
    rfit3.SetParName(0, "constant3")
    rfit3.SetParName(1, "slope3")
    rfit3.SetParameter("slope3", -1.0 / 0.3)
    rfit4.SetParName(0, "constant4")
    rfit4.SetParName(1, "slope4")
    rfit4.SetParameter("slope4", -1.0 / 1.0)
    rfit5.SetParName(0, "constant5")
    rfit5.SetParName(1, "slope5")
    rfit5.SetParameter("slope5", -1.0 / 3.0)
    rfit6.SetParName(0, "constant6")
    rfit6.SetParName(1, "slope6")
    rfit6.SetParameter("slope6", -1.0 / 0.3)

    rfit1_integral = rfit1.Integral(0, 10)
    rfit1.SetParameter(0, rfit1.GetParameter(0) + math.log(1 / rfit1_integral))
    rfit2_integral = rfit2.Integral(0, 10)
    rfit2.SetParameter(0, rfit2.GetParameter(0) + math.log(1 / rfit2_integral))
    rfit3_integral = rfit3.Integral(0, 10)
    rfit3.SetParameter(0, rfit3.GetParameter(0) + math.log(1 / rfit3_integral))
    rfit4_integral = rfit4.Integral(0, 10)
    rfit4.SetParameter(0, rfit4.GetParameter(0) + math.log(1 / rfit4_integral))
    rfit5_integral = rfit5.Integral(0, 10)
    rfit5.SetParameter(0, rfit5.GetParameter(0) + math.log(1 / rfit5_integral))
    rfit6_integral = rfit6.Integral(0, 10)
    rfit6.SetParameter(0, rfit6.GetParameter(0) + math.log(1 / rfit6_integral))

    weight_plot_1 = ROOT.TF1("weight_plot_1", "rfit2*(1/rfit1)")  # 10mm->30mm
    weight_plot_1.SetLineColor(ROOT.kMagenta)
    weight_plot_2 = ROOT.TF1("weight_plot_2", "rfit3*(1/rfit1)")  # 10mm->3mm
    weight_plot_2.SetLineColor(ROOT.kRed)

    weight_plot_3 = ROOT.TF1("weight_plot_3", "rfit5*(1/rfit4)")  # 10mm->30mm
    weight_plot_3.SetLineColor(ROOT.kMagenta)
    weight_plot_4 = ROOT.TF1("weight_plot_4", "rfit6*(1/rfit4)")  # 10mm->3mm
    weight_plot_4.SetLineColor(ROOT.kRed)

    weight_plot_1.SetRange(0.0, 10)
    weight_plot_2.SetRange(0.0, 10)
    weight_plot_3.SetRange(0.0, 10)
    weight_plot_4.SetRange(0.0, 10)

    h_rfdisp_1.Reset()
    h_rfdisp_2.Reset()

    h_lfdisp_1 = ROOT.TH1F(  # Lab Frame dBV LLP1
        "MC Higgs->ss " + tau + " " + m + " GeV ",
        "H->LLP1->4d w/ LLP mass = " + m + "GeV; 3D Decay Length "
                                           "(cm); # Events", 50, 0, 10)
    h_lfdisp_1_reweight_30mm = ROOT.TH1F(  # Lab Frame dBV LLP1 ->30mm
        "MC Higgs->ss " + tau + " " + m + " GeV Reweight -> 30mm",
        "Displacement in Lab Frame (1); 3D Decay Length "
        "(cm); # Events", 50, 0, 10)
    h_lfdisp_1_reweight_3mm = ROOT.TH1F(  # Lab Frame dBV LLP1 ->3mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 3mm",
        "Displacement in Lab Frame (1); 3D Decay Length "
        "(cm); # Events", 50, 0, 10)
    h_lfdisp_2 = ROOT.TH1F(  # Lab Frame dBV LLP2
        "MC Higgs->ss " + tau + " " + m + " GeV",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 3D Decay Length "
                                           "(cm); # Events", 50, 0, 10)
    h_lfdisp_2_reweight_30mm = ROOT.TH1F(  # Lab Frame dBV LLP2 ->30mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 30mm",
        "Displacement in Lab Frame (2); 3D Decay Length "
        "(cm); # Events", 50, 0, 10)
    h_lfdisp_2_reweight_3mm = ROOT.TH1F(  # Lab Frame dBV LLP2 ->3mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 3mm",
        "Displacement in Lab Frame (2); 3D Decay Length "
        "(cm); # Events", 50, 0, 10)

    h_rfdisp_1 = ROOT.TH1F(  # Rest Frame dBV LLP1
        "MC Higgs->ss " + tau + " " + m + " GeV ",
        "H->LLP1->4d w/ LLP mass = " + m + "GeV; ctau "
                                           "(cm); # Events", 20, 0, 5)
    h_rfdisp_1_reweight_30mm = ROOT.TH1F(  # Rest Frame dBV LLP1 ->30mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 30mm",
        "; ctau "
        "(cm); # Events", 20, 0, 5)
    h_rfdisp_1_reweight_3mm = ROOT.TH1F(  # Rest Frame dBV LLP1 ->3mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 3mm",
        "; ctau "
        "(cm); # Events", 20, 0, 5)
    h_newweight_30mm = ROOT.TH1F(  # Rest Frame dBV LLP1 ->100um
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 30mm",
        "; new weights "
        "; # Events", 80, 0, 20)
    h_newweight_3mm = ROOT.TH1F(  # Rest Frame dBV LLP1 ->3mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 3mm",
        "; new weights "
        "; # Events", 80, 0, 20)
    h_rfdisp_2 = ROOT.TH1F(  # Rest Frame dBV LLP2
        "MC Higgs->ss " + tau + " " + m + " GeV ",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 3D Decay Length "
                                           "(cm); # Events", 50, 0, 30)
    h_rfdisp_2_reweight_30mm = ROOT.TH1F(  # Rest Frame dBV LLP2 ->30mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 30mm",
        "Displacement in Rest Frame (2); 3D Decay Length "
        "(cm); # Events", 50, 0, 30)
    h_rfdisp_2_reweight_3mm = ROOT.TH1F(  # Rest Frame dBV LLP2 ->3mm
        "MC Higgs->ss " + tau + " " + m + " GeV Scale -> 3mm",
        "Displacement in Rest Frame (2); 3D Decay Length "
        "(cm); # Events", 50, 0, 30)


    rf1, rf2 = TLorentzVector(), TLorentzVector()
    # plots that are not printed  
    h_rf_dbv_1_reweight_30mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->30mm " + m + " GeV",
        "H->LLP1->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP1 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", 50, 0, 2.5)  # rf
    h_rf_dbv_2_reweight_30mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->30mm " + m + " GeV",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP2 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", 50, 0, 2.5)  # rf
    h_lf_dbv_1_reweight_30mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->30mm " + m + " GeV",
        "H->LLP1->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP1 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", lfbins, 0, 2.5)  # lf
    h_lf_dbv_2_reweight_30mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->30mm " + m + " GeV",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP2 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", lfbins, 0, 2.5)  # lf
    h_rf_dbv_1_reweight_3mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->3mm " + m + " GeV",
        "H->LLP1->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP1 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", 50, 0, 2.5)  # rf
    h_rf_dbv_2_reweight_3mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->3mm " + m + " GeV",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP2 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", 50, 0, 2.5)  # rf
    h_lf_dbv_1_reweight_3mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->3mm " + m + " GeV",
        "H->LLP1->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP1 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", lfbins, 0, 2.5)  # lf
    h_lf_dbv_2_reweight_3mm = ROOT.TH1F(
        "MC Higgs->ss " + tau + "->3mm " + m + " GeV",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP2 Reconstructed "
                                           "Vertex and Beamspot (cm); # Events", lfbins, 0, 2.5)  # lf
    h_lf_dbv_1 = ROOT.TH1F(  # Lab Frame dBV LLP1
        "MC Higgs->ss " + tau + " " + m + " GeV",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP1 Reconstructed Vertex and "
                                           "Beamspot (cm); # Events", lfbins, 0, 2.5)
    h_lf_dbv_2 = ROOT.TH1F(  # Lab Frame dBV LLP2
        "MC Higgs->ss " + tau + " " + m + " GeV",
        "H->LLP2->4d w/ LLP mass = " + m + "GeV; 2D Separation of LLP2 Reconstructed Vertex and "
                                           "Beamspot (cm); # Events", lfbins, 0, 2.5)
    h_weight = ROOT.TH1F(  # Default Weight
        "MC Higgs->ss " + tau + " " + m + " GeV",
        "Weight H->LLP2->4d w/ LLP mass = " + m + "GeV; Weight "
                                                  "; # Events", 50, 0, 30)
    h_weight_to_30mm = ROOT.TH1F(  # To 30mm Weight
        "MC Higgs->ss " + tau + " " + m + " GeV",
        "Weight H->LLP2->4d w/ LLP mass = " + m + "GeV; Weight "
                                                  "; # Events", 50, 0, 30)
    h_weight_to_3mm = ROOT.TH1F(  # To 3mm Weight
        "MC Higgs->ss " + tau + " " + m + " GeV",
        "Weight H->LLP2->4d w/ LLP mass = " + m + "GeV; Weight "
                                                  "; # Events", 50, 0, 30)

    lifetime = 1
    count_t_entry = 0
    for entry in t: #loop over selected events to print ctau and new weight plots
        count_t_entry += 1
        if limit and count_t_entry > 500:
            print("ENTRY LIMIT IS ON!")
            break
        if weight_on :
            evt_weight = entry.weight
        evt_run = entry.run
        evt_bs2derr_0 = entry.bs2derr0
        evt_bs2derr_1 = entry.bs2derr1
        evt_gen_daughters = entry.gen_daughters
        evt_gen_jet_ht30 = entry.gen_jet_ht30 
        gen0, gen1, gen2, gen3 = TLorentzVector(), TLorentzVector(), TLorentzVector(), TLorentzVector()
        lfdisp_1 = ((entry.bsx + entry.gen_x[0] - entry.gen_pv_x0) ** 2 + (
                entry.bsy + entry.gen_y[0] - entry.gen_pv_y0) ** 2 + (
                            entry.bsz + entry.gen_z[0] - entry.gen_pv_z0) ** 2) ** 0.5
        lfdisp_2 = ((entry.bsx + entry.gen_x[1] - entry.gen_pv_x0) ** 2 + (
                entry.bsy + entry.gen_y[1] - entry.gen_pv_y0) ** 2 + (
                            entry.bsz + entry.gen_z[1] - entry.gen_pv_z0) ** 2) ** 0.5
        rf1.SetPtEtaPhiM(entry.gen_lsp_pt[0], entry.gen_lsp_eta[0], entry.gen_lsp_phi[0], entry.gen_lsp_mass[0])
        rf2.SetPtEtaPhiM(entry.gen_lsp_pt[1], entry.gen_lsp_eta[1], entry.gen_lsp_phi[1], entry.gen_lsp_mass[1])
        rfdisp_1 = lfdisp_1 / rf1.Gamma() / rf1.Beta()
        rfdisp_2 = lfdisp_2 / rf2.Gamma() / rf2.Beta()
        weight_10mm_to_30mm = weight_plot_1.Eval(rfdisp_1) * weight_plot_3.Eval(rfdisp_2)
        weight_10mm_to_3mm = weight_plot_2.Eval(rfdisp_1) * weight_plot_4.Eval(rfdisp_2)
        
        if (0.0080 > evt_bs2derr_0 > 0 and 0.0080 > evt_bs2derr_1 > 0 ): 
                h_rfdisp_1.Fill(rfdisp_1, evt_weight)
                h_rfdisp_1_reweight_30mm.Fill(rfdisp_1, evt_weight * weight_10mm_to_30mm)
                h_newweight_30mm.Fill(weight_10mm_to_30mm)
                h_rfdisp_1_reweight_3mm.Fill(rfdisp_1, evt_weight * weight_10mm_to_3mm)
                h_newweight_3mm.Fill(weight_10mm_to_3mm)
                h_cut_gen_jet_ht30.Fill(evt_gen_jet_ht30, evt_weight)
                h_30mm_cut_gen_jet_ht30.Fill(evt_gen_jet_ht30, evt_weight * weight_10mm_to_30mm)
                h_3mm_cut_gen_jet_ht30.Fill(evt_gen_jet_ht30, evt_weight * weight_10mm_to_3mm)
      
    ROOT.gStyle.SetOptStat(112211)
    ROOT.gStyle.SetOptFit(1111)
    err = ROOT.double(0.)
    tot_evt = h_sums.GetBinContent(1)
    err_evt = tot_evt**0.5
    print("total events: %f +/- %f" %(tot_evt,err_evt))
    tot_weighted_evt = h.Integral()
    err_weighted_evt = tot_weighted_evt**0.5
    print("validity check of h_sums by h.Integral() : %f +/- %f" %(tot_weighted_evt,err_weighted_evt)) 

    #events with vertex and event selection applied
    tot_10mm_sel = h_cut_gen_jet_ht30.IntegralAndError(0,201,err, "") 
    alt_rel_err_10mm_sel = err/tot_10mm_sel
    tot_30mm_sel = h_30mm_cut_gen_jet_ht30.IntegralAndError(0,201,err, "")   
    alt_rel_err_30mm_sel = err/tot_30mm_sel
    tot_3mm_sel = h_3mm_cut_gen_jet_ht30.IntegralAndError(0,201,err, "") 
    alt_rel_err_3mm_sel = err/tot_3mm_sel
    sig_eff_10mm = h_cut_gen_jet_ht30.Integral(0,201)/tot_evt 
    sig_eff_30mm = h_30mm_cut_gen_jet_ht30.Integral(0,201)/tot_evt
    sig_eff_3mm = h_3mm_cut_gen_jet_ht30.Integral(0,201)/tot_evt
    
   
    # Table 1.5 & 2.5 in https://www.evernote.com/shard/s376/nl/66335180/cfe155a1-c14c-4946-adb3-a3033881cf75 
    if (prestudy):
        print("** Result below is before fixing the sample size to the desired request and should have event weight + signal eff. off **")
        tot_evt_to_req = tot_evt 
        dp = 1.0
    else :
        print("** Result below is based on a fixed requested sample size of %.2f and an estimate signal eff of %.2f percent  (weight_on=1 is recommended) **" %(tot_evt_to_req, dp*100))
    if (weight_on):
        tot_evt_to_reg = tot_weighted_evt

    # Table 1.2 & 2.2 or 1.5 & 2.5 in https://www.evernote.com/shard/s376/nl/66335180/cfe155a1-c14c-4946-adb3-a3033881cf75
    print("reweighted 10mm->3mm events (rel err): %f +/- %f (+/-%f) " % (sig_eff_3mm*tot_evt_to_req*dp, (sig_eff_3mm*tot_evt_to_req*dp*(alt_rel_err_3mm_sel)*((tot_evt/tot_evt_to_req)**0.5)), round(100*alt_rel_err_3mm_sel*((tot_evt/tot_evt_to_req)**0.5),2)))
    print("nominal 10mm sig events (rel err): %f +/- %f (+/-%f)" % (sig_eff_10mm*tot_evt_to_req*dp, (sig_eff_10mm*tot_evt_to_req*dp*(alt_rel_err_10mm_sel)*((tot_evt/tot_evt_to_req)**0.5)), round(100*alt_rel_err_10mm_sel*((tot_evt/tot_evt_to_req)**0.5),2))) 
    print("reweighted 10mm->30mm events (rel err): %f +/- %f (+/-%f)" % (sig_eff_30mm*tot_evt_to_req*dp, (sig_eff_30mm*tot_evt_to_req*dp*(alt_rel_err_30mm_sel)*((tot_evt/tot_evt_to_req)**0.5)), round(100*alt_rel_err_30mm_sel*((tot_evt/tot_evt_to_req)**0.5),2))) 
        
    # Table 1.4 & 2.4 in https://www.evernote.com/shard/s376/nl/66335180/cfe155a1-c14c-4946-adb3-a3033881cf75
    rel_err = 5.0 
    alt_rel_err_max_sel = max(alt_rel_err_3mm_sel, alt_rel_err_10mm_sel, alt_rel_err_30mm_sel)
    requestevt = tot_evt*((round(alt_rel_err_max_sel*100,2)/rel_err)**2) 
    print("requested events based on largest rel err : %f with %f rel err" %(requestevt,round(100*alt_rel_err_max_sel*((tot_evt/tot_evt_to_req)**0.5),2)))
    print("reweighted 10mm->3mm rel err: %f " % (((tot_evt/requestevt)**0.5)*round(alt_rel_err_3mm_sel*100,2)))
    print("nominal request rel err: %f " % (((tot_evt/requestevt)**0.5)*round(alt_rel_err_10mm_sel*100,2)))
    print("reweighted 10mm->30mm rel err: %f " % (((tot_evt/requestevt)**0.5)*round(alt_rel_err_30mm_sel*100,2))) 

    h_rfdisp_1_reweight_3mm.SetLineColor(ROOT.kRed)
    h_rfdisp_1_reweight_30mm.SetLineColor(ROOT.kMagenta)
    h_rfdisp_1.SetLineColor(ROOT.kBlue)
    h_rfdisp_1_reweight_30mm.Draw()
    h_rfdisp_1_reweight_3mm.Draw("sames")
    h_rfdisp_1.Draw("sames")
    c0.Update()
    st1 = h_rfdisp_1_reweight_3mm.GetListOfFunctions().FindObject("stats")
    st1.SetY1NDC(0.75)
    st1.SetY2NDC(0.55)
    st1.SetLineColor(ROOT.kRed)
    ROOT.gPad.Update()
    c0.Update()
    st2 = h_rfdisp_1.GetListOfFunctions().FindObject("stats")
    st2.SetY1NDC(0.55)
    st2.SetY2NDC(0.35)
    st2.SetLineColor(ROOT.kBlue)
    st3 = h_rfdisp_1_reweight_30mm.GetListOfFunctions().FindObject("stats")
    st3.SetY1NDC(0.35)
    st3.SetY2NDC(0.15)
    st3.SetLineColor(ROOT.kMagenta)
    ROOT.gPad.Update()
    c0.Update()
    c0.Print("gen_rfdisp_reweight_" + tau + "_" + m + "_GeV" + ".png")
    c0.Print("gen_rfdisp_reweight_" + tau + "_" + m + "_GeV" + ".root")
    c0.Clear()
    
    h_newweight_30mm.SetLineColor(ROOT.kMagenta)
    h_newweight_3mm.SetLineColor(ROOT.kRed)
    h_newweight_30mm.Draw()
    h_newweight_3mm.Draw("sames")
    c0.SetLogy()
    c0.Update()
    st1 = h_newweight_3mm.GetListOfFunctions().FindObject("stats")
    st1.SetY1NDC(0.75)
    st1.SetY2NDC(0.55)
    st1.SetLineColor(ROOT.kRed)
    ROOT.gPad.Update()
    c0.Update()
    st3 = h_newweight_30mm.GetListOfFunctions().FindObject("stats")
    st3.SetY1NDC(0.35)
    st3.SetY2NDC(0.15)
    st3.SetLineColor(ROOT.kMagenta)
    ROOT.gPad.Update()
    c0.Update()
    c0.Print("newweight_" + tau + "_" + m + "_GeV" + ".png")
    c0.Print("newweight_" + tau + "_" + m + "_GeV" + ".root") 
    fA.Close()
print("Done!")
