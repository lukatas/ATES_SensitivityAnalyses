"""
Created on Tue May 16 09:08:19 2023

@author: lhtas

script for running imported modelmuse models (MF5) prepared for parallel running
"""

import os
import pandas as pd
import numpy as np
import flopy
import flopy.modflow as fpm
import flopy.mt3d as fpt
from ATES_SensitivityAnalyses.Case_2.Parallel_Simulations.utils import (
    Q_sine,
    T_sine,
    NewGridValues,
)
from ATES_SensitivityAnalyses.Case_2.Parallel_Simulations.T_topAqf import T_TopAqf

def FlowTransport(
    exe_mf: str,
    exe_mt: str,
    sim_ws: str,
    Kh_aqf1: float,
    Kh_aqf2: float,
    Kv_aqf1: float,
    Kv_aqf2: float,
    gradient: float,
    por_Taqf: float,
    por_Eaqf: float,
    volume: float,
    longitudinal: float,
    Twinter: float,
    Tzomer: float,
    Rch: float,
    results_dir: str,
):
    """load flow and transport model"""
    f = os.path.join(sim_ws, "Luik_2Y.nam")
    mf = flopy.modflow.Modflow.load(
        f, version="mf2005", exe_name=exe_mf, model_ws=sim_ws
    )

    f = "Luik_2Y.mt_nam"
    # problem with loading rch package so do not include it
    mt = fpt.Mt3dms.load(
        f,
        model_ws=sim_ws,
        version="mt3d-usgs",
        exe_name=exe_mt,
        modflowmodel=mf,
        load_only=["btn", "adv", "dsp", "rct", "gcg"],
        verbose=True,
    )
    # ! need to remove first line of .dsp file

    """ create Xarrays """
    XGR = mf.dis.delr.array
    YGR = mf.dis.delc.array
    layers = mf.dis.nlay

    """ change directories for simulations """
    mf.change_model_ws(results_dir)
    mt.change_model_ws(results_dir)

    """ save variable parameters """
    parameters = {
        "Kh_aqf1": [Kh_aqf1],
        "Kh_aqf2": [Kh_aqf2],
        "Kv_aqf1": [Kv_aqf1],
        "Kv_aqf2": [Kv_aqf2],
        "gradient": [gradient],
        "por_Taqf": [por_Taqf],
        "por_Eaqf": [por_Eaqf],
        "volume": [volume],
        "longitudinal": [longitudinal],
        "Twinter": [Twinter],
        "Tzomer": [Tzomer],
        "Recharge": [Rch],
    }
    parameters = pd.DataFrame(data=parameters)
    run_name = os.path.basename(results_dir)
    parameters.to_csv(os.path.join(results_dir, "Parameters_{}.csv".format(run_name)))

    # Dis
    # need to set unitnumber to 12 if problem
    botm = mf.dis.botm.array
    nlay = layers
    top = mf.dis.top.array
    nrow = mf.dis.nrow
    ncol = mf.dis.ncol
    nper = mf.dis.nper
    delr = XGR
    delc = YGR
    laycbd = mf.dis.laycbd.array
    perlen = mf.dis.perlen.array
    nstp = mf.dis.nstp.array
    tsmult = mf.dis.tsmult.array
    steady = mf.dis.steady.array
    itmuni = mf.dis.itmuni
    lenuni = mf.dis.lenuni
    extension = "dis"
    unitnumber = None
    filenames = None
    start_datetime = mf.dis.start_datetime
    rotation = 0.0

    fpm.ModflowDis(
        mf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        delr=delr,
        delc=delc,
        laycbd=laycbd,
        top=top,
        botm=botm,
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        steady=steady,
        itmuni=itmuni,
        lenuni=lenuni,
        extension=extension,
        unitnumber=12,
        filenames=filenames,
        start_datetime=start_datetime,
        rotation=rotation,
    )

    # Lpf
    laytyp = mf.lpf.laytyp
    layavg = mf.lpf.layavg
    chani = mf.lpf.chani
    layvka = mf.lpf.layvka
    laywet = mf.lpf.laywet
    ipakcb = mf.lpf.ipakcb  # flag used to determine of cbc budget data should be saved (non zero: budget will be saved)
    hdry = mf.lpf.hdry
    iwdflg = 0
    wetfct = mf.lpf.wetfct
    iwetit = mf.lpf.iwetit
    ihdwet = mf.lpf.ihdwet
    hani = mf.lpf.hani
    vka = mf.lpf.vka.array
    ss = mf.lpf.ss
    sy_new = por_Eaqf
    vkcb = mf.lpf.vkcb
    wetdry = mf.lpf.wetdry
    storagecoefficient = False
    constantcv = False
    thickstrt = False
    nocvcorrection = False
    novfc = False
    hk = mf.lpf.hk.array
    extension = "lpf"
    unitnumber = None
    filenames = None

    top_aqf = 59
    botm_aqf = 49
    aqf_layers = []
    for lay in range(len(botm)):
        # assumes horizontal aqf layers (check value of first cel)
        if botm[lay][0][0] < top_aqf and botm[lay][0][0] >= botm_aqf:
            aqf_layers.append(lay)

    botm_aqf1 = 54
    hk_new = hk
    vka_new = vka
    for lay in aqf_layers:
        if botm[lay][0][0] >= botm_aqf1:
            hk_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kh_aqf1)
            vka_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kv_aqf1)
        else:
            hk_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kh_aqf2)
            vka_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kv_aqf2)

    fpm.ModflowLpf(
        mf,
        laytyp=laytyp,
        layavg=layavg,
        chani=chani,
        layvka=layvka,
        laywet=laywet,
        ipakcb=ipakcb,
        hdry=hdry,
        iwdflg=iwdflg,
        wetfct=wetfct,
        iwetit=iwetit,
        ihdwet=ihdwet,
        hani=hani,
        vka=vka_new,
        ss=ss,
        sy=sy_new,
        vkcb=vkcb,
        wetdry=wetdry,
        storagecoefficient=storagecoefficient,
        constantcv=constantcv,
        thickstrt=thickstrt,
        nocvcorrection=nocvcorrection,
        novfc=novfc,
        hk=hk_new,
        extension=extension,
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # Chd
    # set new gradient based on original values
    NB = 63
    SB = 62
    change_original = NB - SB
    grad_original = ((change_original) / 1000) * 100

    # new Northern and southern boundary conditions
    grad_new = gradient
    change_new = grad_new * (1000 / 100)  # 1000 is distance between boundaries
    if grad_new >= grad_original:
        NB = NB + (change_new - change_original) / 2
        SB = SB - (change_new - change_original) / 2
    else:
        NB = NB - (change_original - change_new) / 2
        SB = SB + (change_original - change_new) / 2

    # write chd_data
    chd_data = {}
    NSB = [0, nrow - 1]  # row of northern and southern boundary, zero-based

    for stp in range(nper):
        chd_data[stp] = []
        for k in range(nlay):
            for i in range(nrow):
                if i in NSB:
                    for j in range(ncol):
                        if i == NSB[0]:
                            chd_data[stp].append((k, i, j, NB, NB))
                        else:
                            chd_data[stp].append((k, i, j, SB, SB))

    # create new package
    fpm.ModflowChd(
        mf,
        stress_period_data=chd_data,
        dtype=None,
        extension="chd",
        unitnumer=unitnumber,
        filenames=None,
    )

    # Wel
    # write BC
    ipakcb = mf.wel.ipakcb
    well_data = {}
    EB = 0
    WB = 0
    EWB = [0, ncol - 1]  # row of eastern and western boundary, zero-based

    for stp in range(nper):
        well_data[stp] = []
        for k in range(nlay):
            for i in range(
                ncol
            ):  # not first and last row, we already have CHD BC there
                if i in EWB:
                    for j in range(nrow - 1):
                        if i == EWB[0] and j > 0:
                            well_data[stp].append((k, j, i, WB))
                        elif i == EWB[-1] and j > 0:
                            well_data[stp].append((k, j, i, EB))

    # wells
    # change flowrate per layer according to new Kh value
    # get the well locations (lay, row, col) in the zero-based grid (>< lambert coordinates) !condition: all wells already active in first stress period!
    wells = mf.wel.stress_period_data.data
    wwell = []
    cwell = []
    for x in range(len(wells[0])):  # warm wells start injecting
        if wells[0][x][3] > 0:
            points = []
            for crds in range(len(wells[0][x])):
                if crds < 3:
                    points.append(wells[0][x][crds])

            wwell.append(points)

        elif wells[0][x][3] < 0:
            points = []
            for crds in range(len(wells[0][x])):
                if crds < 3:
                    points.append(wells[0][x][crds])
            cwell.append(points)

    # flowrate changes every stress period
    Dis_aqf = 1  # thickness of layers in aqf

    # thickness of each aquifer
    L1 = top_aqf - botm_aqf1
    L2 = botm_aqf1 - botm_aqf

    # transmissivity of aqf
    T1 = L1 * Kh_aqf1
    T2 = L2 * Kh_aqf2
    Ttot = T1 + T2

    F1 = T1 / Ttot
    F2 = T2 / Ttot

    # this is the total flowrate for a wel per stress period, still need to divide if well is divided over several cells
    Q = Q_sine(
        vol=volume,
        time_dis_sine=perlen[0] / 24 / 3600,
        sim_time_years=(len(perlen) * perlen[0]) / 24 / 3600 / 360,
        year_days=360,
    )

    Q_wwel_aqf1 = [(i * F1) / (L1 / Dis_aqf) for i in Q]
    Q_wwel_aqf2 = [(i * F2) / (L2 / Dis_aqf) for i in Q]

    wwell_rate_perlay = {}
    for stp in range(nper):
        wwell_rate_perlay[stp] = []
        for lay in aqf_layers:
            if botm[lay][0][0] >= botm_aqf1:
                wwell_rate_perlay[stp].append(Q_wwel_aqf1[stp])
            else:
                wwell_rate_perlay[stp].append(Q_wwel_aqf2[stp])

    for stp in range(nper):
        if (
            wwell_rate_perlay[stp][0] > 0
        ):  # when rate wwell is postive, injection in wwel, extraction in cold well
            for well in range(len(wwell)):
                well_data[stp].append(
                    (
                        wwell[well][0],
                        wwell[well][1],
                        wwell[well][2],
                        wwell_rate_perlay[stp][well],
                    )
                )
            for well in range(len(cwell)):
                well_data[stp].append(
                    (
                        cwell[well][0],
                        cwell[well][1],
                        cwell[well][2],
                        wwell_rate_perlay[stp][well] * -1,
                    )
                )
        else:
            for well in range(len(wwell)):
                well_data[stp].append(
                    (
                        wwell[well][0],
                        wwell[well][1],
                        wwell[well][2],
                        wwell_rate_perlay[stp][well],
                    )
                )
            for well in range(len(cwell)):
                well_data[stp].append(
                    (
                        cwell[well][0],
                        cwell[well][1],
                        cwell[well][2],
                        wwell_rate_perlay[stp][well] * -1,
                    )
                )

    fpm.ModflowWel(
        mf,
        ipakcb=ipakcb,
        stress_period_data=well_data,
        dtype=None,
        extension="wel",
        unitnumber=None,
        filenames=None,
    )
    # Rch
    nrchop = mf.rch.nrchop
    ipakcb = mf.rch.ipakcb
    rech = Rch  # mf.rch.rech.array[0][0][0][0]
    irch = mf.rch.irch

    fpm.ModflowRch(
        mf,
        nrchop=nrchop,
        ipakcb=ipakcb,
        rech=rech,
        irch=irch,
        extension="rch",
        unitnumber=unitnumber,
        filenames=None,
    )

    # Hob,Bas6 package are written correctly
    # define starting head in every layer according to gradient.. otherwise not
    # correct for layers with very low hydraulic conductivity where flow
    # is too slow to establish gradient (don't have steady state period)

    distance = YGR.cumsum()
    strt = np.ones((nlay, nrow, ncol))
    for lay in range(nlay):
        for row in range(len(distance)):
            row_value = NB - ((grad_new / 100) * distance[row])
            strt[lay][row] = strt[lay][row] * row_value

    # Bas6
    ibound = mf.bas6.ibound.array
    ichflg = mf.bas6.ichflg
    stoper = mf.bas6.stoper
    hnoflo = mf.bas6.hnoflo
    fpm.ModflowBas(
        mf,
        ibound=ibound,
        strt=strt,
        ichflg=ichflg,
        stoper=stoper,
        hnoflo=hnoflo,
        extension="bas",
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # write and run flow model
    mf.write_input()
    mf.run_model(silent=False, report=True)

    # Btn
    #     #want to be able to multiply output of cbc file by values of ucn file at these times
    #     #therefore need to save results at these times as well (so every half day (as we want for mto file) and everytime we have output for cbc)

    MFStyleArr = False
    DRYCell = False
    Legacy99Stor = False
    FTLPrint = False
    NoWetDryPrint = False
    OmitDryBud = False
    AltWTSorb = False
    ncomp = mt.btn.ncomp  # The number of components
    mcomp = ncomp
    tunit = mt.btn.tunit
    lunit = mt.btn.lunit
    munit = mt.btn.munit
    prsity = mt.btn.prsity.array
    icbund = mt.btn.icbund.array
    sconc = mt.btn.sconc[0].array  # Starting concentration
    cinact = mt.btn.cinact
    thkmin = mt.btn.thkmin
    ifmtcn = mt.btn.ifmtcn
    ifmtnp = mt.btn.ifmtnp
    ifmtrf = mt.btn.ifmtrf
    ifmtdp = mt.btn.ifmtdp
    savucn = True  # Save concentration array or not
    nprs = mt.btn.nprs  # when saved, how many times
    timprs = mt.btn.timprs  # times saved,  number of entries = nprs
    obs = mt.btn.obs
    nprobs = mt.btn.nprobs  # frequence save conc at obs
    chkmas = True
    nprmas = mt.btn.nprmas
    perlen = perlen
    nstp = nstp  # steps per stress period
    tsmult = tsmult  # multiplier for time steps
    ssflag = None
    dt0 = mt.btn.dt0.array[0]
    mxstrn = mt.btn.mxstrn.array[0]
    ttsmult = tsmult[
        0
    ]  # The multiplier for successive transport steps within a flow time-step
    ttsmax = mt.btn.ttsmax.array[0]
    extension = "btn"
    unitnumber = None
    filenames = None

    prsity_new = prsity
    for lay in aqf_layers:
        prsity_new[lay] = NewGridValues(nrow, ncol, por_Eaqf)

    # write new Btn package
    flopy.mt3d.mtbtn.Mt3dBtn(
        model=mt,
        MFStyleArr=MFStyleArr,
        DRYCell=DRYCell,
        Legacy99Stor=Legacy99Stor,
        FTLPrint=FTLPrint,
        NoWetDryPrint=NoWetDryPrint,
        OmitDryBud=OmitDryBud,
        AltWTSorb=AltWTSorb,
        ncomp=ncomp,
        mcomp=mcomp,
        tunit=tunit,
        lunit=lunit,
        munit=munit,
        prsity=prsity_new,
        icbund=icbund,
        sconc=sconc,
        cinact=cinact,
        thkmin=thkmin,
        ifmtcn=ifmtcn,
        ifmtnp=ifmtnp,
        ifmtrf=ifmtrf,
        ifmtdp=ifmtdp,
        savucn=savucn,
        nprs=nprs,
        timprs=timprs,
        obs=obs,
        nprobs=nprobs,
        chkmas=chkmas,
        nprmas=nprmas,
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        ssflag=ssflag,
        dt0=dt0,
        mxstrn=mxstrn,
        ttsmult=ttsmult,
        ttsmax=ttsmax,
        extension=extension,
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # Ssm (stress period data worden niet correct geladen)
    cevt = None
    mxss = None  # If problem, set to 10000
    dtype = None
    extension = "ssm"
    unitnumber = None
    filenames = None

    ssm_data = {}
    itype = {
        "BAS6": 1,
        "CC": -1,
        "CHD": 1,
        "DRN": 3,
        "GHB": 5,
        "MAS": 15,
        "PBC": 1,
        "RIV": 4,
        "WEL": 2,
    }

    T_sin = T_sine(
        Twinter=Twinter,
        Tzomer=Tzomer,
        time_dis_sine=perlen[0] / 24 / 3600,
        sim_time_years=(len(perlen) * perlen[0]) / 24 / 3600 / 360,
        year_days=360,
    )

    # ssm for boundaries
    temp_initial = 10
    for stp in range(nper):
        ssm_data[stp] = []
        for k in range(nlay):
            # impose T top boundary
            if k == 0:
                for i in range(nrow):
                    for j in range(ncol):
                        ssm_data[stp].append(
                            (k, i, j, T_sin[stp], itype["CC"])
                        )  # all rows and colums of top layer get T_sin

            for i in range(nrow):
                if i in NSB:
                    for j in range(ncol):
                        ssm_data[stp].append((k, i, j, temp_initial, itype["CC"]))
                        ssm_data[stp].append((k, i, j, temp_initial, itype["CHD"]))
                else:  # dit zijn dan oost en west boundaries
                    ssm_data[stp].append((k, i, 0, temp_initial, itype["CC"]))
                    ssm_data[stp].append((k, i, 0, temp_initial, itype["WEL"]))
                    ssm_data[stp].append((k, i, ncol - 1, temp_initial, itype["CC"]))
                    ssm_data[stp].append((k, i, ncol - 1, temp_initial, itype["WEL"]))

    # ssm for wells
    warm = 15
    cold = 5
    for stp in range(nper):
        if (
            wwell_rate_perlay[stp][0] > 0
        ):  # wwell injection: warm T for these stp, wwell extr: cold T
            for well in range(len(wwell)):
                ssm_data[stp].append(
                    (wwell[well][0], wwell[well][1], wwell[well][2], warm, itype["CC"])
                )
                ssm_data[stp].append(
                    (wwell[well][0], wwell[well][1], wwell[well][2], warm, itype["WEL"])
                )
        else:
            for well in range(len(cwell)):
                ssm_data[stp].append(
                    (cwell[well][0], cwell[well][1], cwell[well][2], cold, itype["CC"])
                )
                ssm_data[stp].append(
                    (cwell[well][0], cwell[well][1], cwell[well][2], cold, itype["WEL"])
                )

    # ssm for top boundary, combination with recharge
    crch = {}
    for stp in range(nper):
        crch[stp] = []
        crch[stp].append(T_sin[stp] * np.ones((nrow, ncol), dtype=float))

    # write new Ssm package
    flopy.mt3d.Mt3dSsm(
        model=mt,
        crch=crch,
        cevt=cevt,
        mxss=mxss,
        stress_period_data=ssm_data,
        dtype=dtype,
        extension=extension,
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # Dsp

    k0_aqf = 0.58 * por_Taqf + 3 * (1 - por_Taqf)  # bulk thermal cond = k𝒘𝜽+ 𝒌𝒔(𝟏 −𝜽)
    mol_diff_aqf = k0_aqf / (por_Taqf * 1000 * 4183)  # k0/(𝜽*rhow*cw)

    dmcoef_new = np.zeros((nlay, 1))  # moet shape (nlay,1) zijn
    dmcoef = mt.dsp.dmcoef[0].array
    for lay in range(nlay):
        if lay in aqf_layers:
            dmcoef_new[lay][0] = mol_diff_aqf
        else:
            dmcoef_new[lay][0] = dmcoef[lay]

    al = mt.dsp.al.array
    al_new = al
    for lay in aqf_layers:
        al_new[lay] = NewGridValues(nrow, ncol, longitudinal)

    trpt = mt.dsp.trpt.array  # ratio hor/long disperion: typically 0.1, do not change
    trpv = mt.dsp.trpv.array  # ratio vert/long dispersion: typically 0.01 do not change
    extension = "dsp"
    multiDiff = False
    unitnumber = None
    filenames = None

    # write new Dsp package
    flopy.mt3d.Mt3dDsp(
        model=mt,
        al=al_new,
        trpt=trpt,
        trpv=trpv,
        dmcoef=dmcoef_new,
        extension=extension,
        multiDiff=multiDiff,
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # Rct

    rhob_aqf = 2650 * (
        1 - por_Taqf
    )  # bulk density = density solid * (1-total porosity)

    # therm_distr_aqf = (
    #     878 / (4183 * 1000)
    # )  # thermal distribution coefficient = spec heat capacity solid/(that of water*density water)

    extension = "rct"
    isothm = mt.rct.isothm  # linear
    ireact = mt.rct.ireact
    igetsc = mt.rct.igetsc
    rhob = mt.rct.rhob.array  # bulk density = density solid * (1-total porosity)
    prsity2 = None
    srconc = None
    sp1 = mt.rct.sp1[0].array  # thermal distribution coefficient
    sp2 = mt.rct.sp2[0].array  # read out but not used when isothm is 1
    rc1 = None
    rc2 = None

    rhob_new = rhob
    for lay in aqf_layers:
        rhob_new[lay] = NewGridValues(nrow, ncol, rhob_aqf)

    # write new Rct package
    flopy.mt3d.Mt3dRct(
        model=mt,
        isothm=isothm,
        ireact=ireact,
        igetsc=igetsc,
        rhob=rhob_new,
        prsity2=prsity2,
        srconc=srconc,
        sp1=sp1,
        sp2=sp2,
        rc1=rc1,
        rc2=rc2,
        extension=extension,
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # Adv
    mixelm = 1  # -1: TVD, 1: MOC
    percel = mt.adv.percel  # courant number
    nadvfd = 1  # mt.adv.nadvfd
    npmin = 16  # mt.adv.npmin
    npmax = 32  # mt.adv.npmax
    mxpart = 75000000  # mt.adv.mxpart
    dceps = 1e-005
    nplane = 0
    npl = 0
    nph = 10
    extension = "adv"
    unitnumber = None
    filenames = None

    flopy.mt3d.Mt3dAdv(
        model=mt,
        mixelm=mixelm,
        percel=percel,
        npmin=npmin,
        npmax=npmax,
        mxpart=mxpart,
        nadvfd=nadvfd,
        dceps=dceps,
        nplane=nplane,
        npl=npl,
        nph=nph,
        extension=extension,
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # write and run transport model
    mt.write_input(check=True)
    mt.run_model(report=True)

    # save deltaT and cbc flow for each run
    os.rename(
        os.path.join(results_dir, "Luik_2Y_temperature.ucn"),
        os.path.join(results_dir, "UCN_{}.ucn".format(run_name)),
    )
    os.remove(os.path.join(results_dir, "Luik_2Y_temperature_S.ucn"))
    os.rename(
        os.path.join(results_dir, "Luik_2Y.cbc"),
        os.path.join(results_dir, "CBC_{}.cbc".format(run_name)),
    )

    # mto to csv
    mto_file = "MTO_{}.csv".format(run_name)  # save MTO file for each model run
    T = fpt.mt.Mt3dms.load_obs(os.path.join(results_dir, "Luik_2Y_temperature.mto"))
    os.remove(os.path.join(results_dir, "Luik_2Y_temperature.mto"))

    nlay_aqf = len(aqf_layers)
    column_names = ["Step", "Time(s)"]
    for lay in range(nlay_aqf):
        column_names.append("w{}".format(lay))
        column_names.append("c{}".format(lay))
    T = pd.DataFrame(T)
    T.columns = column_names
    T["Time (d)"] = T["Time(s)"] / 86400

    # mean Tdifference
    T["MeanTw"] = T.filter(regex="^w").sum(axis=1) / nlay_aqf
    T["MeanTc"] = T.filter(regex="^c").sum(axis=1) / nlay_aqf

    T["DeltaT"] = T["MeanTw"] - T["MeanTc"]
    T["DeltaT_cold"] = temp_initial - T["MeanTc"]
    T["DeltaT_warm"] = T["MeanTw"] - temp_initial
    T.to_csv(os.path.join(results_dir, mto_file))

    integral_power_by_season, ucn_xr = T_TopAqf(
        ucn_file=os.path.join(results_dir, "UCN_{}.ucn".format(run_name)),
        cbc_file=os.path.join(results_dir, "CBC_{}.cbc".format(run_name)),
        layer_of_interest=9,
        gwT_natural=temp_initial,
    )

    ucn_xr.to_netcdf(path=os.path.join(results_dir, "UCN_XR_{}.nc".format(run_name)))
    integral_power_by_season.to_netcdf(
        path=os.path.join(results_dir, "EXCHANGE_{}.nc".format(run_name))
    )

    return mf, mt
