import os
import pandas as pd
import numpy as np
import flopy
import flopy.modflow as fpm
import flopy.mt3d as fpt
from utils import NewGridValues

def FlowTransport(exe_mf: str,
                  exe_mt: str,
                  sim_ws: str,
                  results_dir: str,
                  Kh_aqf: float,
                  Kh_aqt: float,
                  Kv_aqf: float,
                  Kv_aqt: float,
                  gradient: float,
                  por_Taqf: float,
                  por_Taqt: float,
                  por_Eaqf: float,
                  por_Eaqt: float,
                  longitudinal: float,
                  aqf_dz: float,
                  deltaT_inj: float,
                  flowrate: float
                  ):
    ''' load flow and transport model '''
    f = os.path.join(sim_ws, 'Campus_MNW_60.nam')
    mf = flopy.modflow.Modflow.load(f, version='mf2005', exe_name=exe_mf, model_ws=sim_ws, load_only = ['lpf','dis','bas6','chd','oc','pcg','lmt6','rch'])
    #hob could be added here, make sure everything from original nam file is here but cannot load MNW!
    #! to load .ob_hob file you need to modify the original file so that the lines for the observation layers are not split over several rows of the file
    #I had 40 obs layers and by default ModelMuse writes 10 observations per line but flopy cannot read it if not all observation layers are not on the same line ;)
    #Need to remove .mnwi_out files from nam file (otherwise runparallel is confused because different name in flopy, don't need those files)

    f = 'Campus_MNW_60.mt_nam'
    mt = fpt.Mt3dms.load(f, model_ws=sim_ws, version='mt3d-usgs', exe_name=exe_mt, modflowmodel=mf,load_only=['btn','adv','dsp','rct','gcg'],verbose=True)
    # ! need to remove first line of .dsp file
    #ftl file does not need to be loaded (created by modflow lmt6 and mt3d finds this output file automatically when running)
    #Need to remove .mt_mnw_out files from nam file (otherwise runparallel error 'cannot access already in use', don't need those files)

    XGR = mf.dis.delr.array
    YGR = mf.dis.delc.array
    layers = mf.dis.nlay

    ''' change directories for simulations '''

    mf.change_model_ws(results_dir)
    mt.change_model_ws(results_dir)

    ''' save variable parameters '''
    parameters = ({'Kh_aqf': [Kh_aqf],
                   'Kh_aqt': [Kh_aqt],
                   'Kv_aqf': [Kv_aqf],
                   'Kv_aqt': [Kv_aqt],
                   'gradient': [gradient],
                   'por_Taqf': [por_Taqf],
                   'por_Taqt': [por_Taqt],
                   'por_Eaqf': [por_Eaqf],
                   'por_Eaqt': [por_Eaqt],
                   'longitudinal': [longitudinal],
                   'aqf_dz': [aqf_dz],
                 #  'deltaT_inj': [deltaT_inj],
                   'flowrate' : [flowrate]
                   })

    parameters = pd.DataFrame(data=parameters)
    run_name = os.path.basename(results_dir)
    parameters.to_csv(os.path.join(results_dir, 'Parameters_{}.csv'.format(run_name)))

# Dis
    # thickness
    botm = mf.dis.botm.array
    top = mf.dis.top.array
    aqf_dz_or = abs(botm[20] - botm[10])[0][0] + abs(botm[8] - botm[3])[0][0] + abs(botm[30] - botm[26])[0][0] # or we know it is 18.5
    nlay = layers

    Yd6 = list(range(4,8+1))
    Yd5 = list(range(9,10+1))
    Yd4 = list(range(11,20+1))
    Yd3 = list(range(21,26+1))
    Yd2 = list(range(27,30+1))
    Yd1 = list(range(31,43+1)) #three lower layers not included (they are modelled to account for downward conduction

    #maintain discretization (every layer of aquifer approximately same thickness)
    number_aqf_lay = len(Yd6) + len(Yd4) + len(Yd2)
    number_aqt_lay = len(Yd5) + len(Yd3) + len(Yd1)

    botm_new = botm
    if aqf_dz >= aqf_dz_or:
        diff = aqf_dz - aqf_dz_or

        change_botm_aqf = diff/number_aqf_lay
        change_botm_aqt = diff/number_aqt_lay

        multipl = 1
        for lay in Yd6:
            botm_new[lay] = botm_new[lay] - (change_botm_aqf*multipl)
            multipl+=1

        multipl = 1
        for lay in Yd5:
            botm_new[lay] = botm_new[lay] - (len(Yd6)*change_botm_aqf) + (change_botm_aqt*multipl)
            multipl+=1

        multipl=1
        for lay in Yd4:
            botm_new[lay] = botm_new[lay] - (len(Yd6)*change_botm_aqf) + (len(Yd5)*change_botm_aqt) - (change_botm_aqf*multipl)
            multipl+=1

        multipl = 1
        for lay in Yd3:
            botm_new[lay] = botm_new[lay] - (len(Yd6) * change_botm_aqf) + (len(Yd5) * change_botm_aqt) - (
                        len(Yd4) * change_botm_aqf) + (change_botm_aqt * multipl)
            multipl += 1

        multipl = 1
        for lay in Yd2:
            botm_new[lay] = botm_new[lay] - (len(Yd6) * change_botm_aqf) + (len(Yd5) * change_botm_aqt) - (
                        len(Yd4) * change_botm_aqf) + (len(Yd3) * change_botm_aqt) - (change_botm_aqf * multipl)
            multipl += 1

        multipl = 1
        for lay in Yd1:
            botm_new[lay] = botm_new[lay] - (len(Yd6) * change_botm_aqf) + (len(Yd5) * change_botm_aqt) - (
                        len(Yd4) * change_botm_aqf) + (len(Yd3) * change_botm_aqt) - (len(Yd2) * change_botm_aqf)+(change_botm_aqt * multipl)
            multipl += 1

    else:
        diff = aqf_dz_or - aqf_dz
        change_botm_aqf = diff/number_aqf_lay
        change_botm_aqt = diff/number_aqt_lay

        multipl = 1
        for lay in Yd6:
            botm_new[lay] = botm_new[lay] + (change_botm_aqf*multipl)
            multipl+=1

        multipl = 1
        for lay in Yd5:
            botm_new[lay] = botm_new[lay] + (len(Yd6)*change_botm_aqf) - (change_botm_aqt*multipl)
            multipl+=1

        multipl=1
        for lay in Yd4:
            botm_new[lay] = botm_new[lay] + (len(Yd6)*change_botm_aqf) - (len(Yd5)*change_botm_aqt) + (change_botm_aqf*multipl)
            multipl+=1

        multipl = 1
        for lay in Yd3:
            botm_new[lay] = botm_new[lay] + (len(Yd6) * change_botm_aqf) - (len(Yd5) * change_botm_aqt) + (
                        len(Yd4) * change_botm_aqf) - (change_botm_aqt * multipl)
            multipl += 1

        multipl = 1
        for lay in Yd2:
            botm_new[lay] = botm_new[lay] + (len(Yd6) * change_botm_aqf) - (len(Yd5) * change_botm_aqt) + (
                        len(Yd4) * change_botm_aqf) - (len(Yd3) * change_botm_aqt) + (change_botm_aqf * multipl)
            multipl += 1

        multipl = 1
        for lay in Yd1:
            botm_new[lay] = botm_new[lay] + (len(Yd6) * change_botm_aqf) - (len(Yd5) * change_botm_aqt) + (
                        len(Yd4) * change_botm_aqf) - (len(Yd3) * change_botm_aqt) + (len(Yd2) * change_botm_aqf) - (change_botm_aqt * multipl)
            multipl += 1


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
        botm=botm_new,
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
    ipakcb = mf.lpf.ipakcb #flag used to determine of cbc budget data should be saved (non zero: budget will be saved)
    hdry = mf.lpf.hdry
    iwdflg = 0
    wetfct = mf.lpf.wetfct
    iwetit = mf.lpf.iwetit
    ihdwet = mf.lpf.ihdwet
    hani = mf.lpf.hani
    vka = mf.lpf.vka.array
    ss = mf.lpf.ss
    sy = mf.lpf.sy
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

    aqf_layers = []
    aqf_layers.extend(Yd6)
    aqf_layers.extend(Yd4)
    aqf_layers.extend(Yd2)
    aqt_layers = []
    aqt_layers.extend(Yd5)
    aqt_layers.extend(Yd3)
    aqt_layers.extend(Yd1)

    hk_new = hk
    vka_new = vka
    sy_new = sy

    for lay in aqf_layers:
        hk_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kh_aqf)
        vka_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kv_aqf)
        sy_new[lay] = NewGridValues(nrow, ncol, por_Eaqf)

    for lay in aqt_layers:
        hk_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kh_aqt)
        vka_new[lay] = NewGridValues(nrow=nrow, ncol=ncol, new_value=Kv_aqt)
        sy_new[lay] = NewGridValues(nrow,ncol,por_Eaqt)

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
        filenames=filenames)

# Chd
    # gradient laten varieren rond originele
    NB = 6.14
    SB = 7.4
    change_original = SB - NB
    grad_original = ((change_original) / 1000) * 100

    # new Northern and southern boundary conditions
    grad_new = gradient
    change_new = grad_new * (1000 / 100)  # 1000 is distance between boundaries
    if grad_new >= grad_original:
        NB = NB - (change_new - change_original) / 2
        SB = SB + (change_new - change_original) / 2
    else:
        NB = NB + (change_original - change_new) / 2
        SB = SB - (change_original - change_new) / 2

    # write chd_data
    chd_data = {}

    NSB = [0, nrow - 1]  # row of northern and southern boundary, zero-based
    EWB = [0, ncol - 1]  # col of eastern and western boundary, zero-based

    for stp in range(nper):
        chd_data[stp] = []
        for k in range(nlay):
            for i in range(nrow):
                if i in NSB:
                    for j in range(ncol):
                        if i == NSB[0]:
                            chd_data[stp].append((k, i, j, NB, NB)) #lay,row,col,shead,ehead
                        else:
                            chd_data[stp].append((k, i, j, SB, SB))

             #establish gradient through E,W boundary ass well because no steady state period
            for i in range(ncol):  # ook niet eerste en laatste rij want daar hebben we al Chd!
                if i in EWB:
                    for j in range(nrow - 1):
                        if i == EWB[0] and j > 0:
                            value = ((change_new/1000) * YGR.cumsum()[j]) + NB
                            chd_data[stp].append((k, j, i, value, value))
                        elif i == EWB[-1] and j > 0:
                            value = ((change_new / 1000) * YGR.cumsum()[j]) + NB
                            chd_data[stp].append((k, j, i, value, value))



    # create new package
    fpm.ModflowChd(
        mf,
        stress_period_data=chd_data,
        dtype=None,
        extension="chd",
        unitnumer=unitnumber,
        filenames=None)


# MNW2
    # flopy is zero based-Modelmuse is not, choose right col and row number (zero based)!
    wwell_i = 42
    wwell_j = 84
    cwell_i = 42
    cwell_j = 42

    # node_data_dtype_list = fpm.ModflowMnw2.get_empty_node_data().dtype.descr
    mnw_columns = ['i', 'j', 'ztop', 'zbotm', 'wellid', 'losstype', 'pumploc', 'qlimit', 'ppflag', 'pumpcap', 'rw']
    mnw_data = [[cwell_i, cwell_j, botm_new[3][0][0] - 0.001, botm_new[Yd6[-1]][0][0] + 0.001, 'cwell', 'thiem', 0, 0, 0, 0, 0.0625],
                [cwell_i, cwell_j, botm_new[Yd5[-1]][0][0] - 0.001, botm_new[Yd4[-1]][0][0]+0.001, 'cwell', 'thiem', 0, 0, 0, 0, 0.0625],
                [cwell_i, cwell_j, botm_new[Yd3[-1]][0][0] - 0.001, botm_new[Yd2[-1]][0][0]+0.001, 'cwell', 'thiem', 0, 0, 0, 0, 0.0625],
                [wwell_i, wwell_j, botm_new[3][0][0] - 0.001, botm_new[Yd6[-1]][0][0] + 0.001, 'wwell', 'thiem', 0, 0, 0, 0, 0.0625],
                [wwell_i, wwell_j, botm_new[Yd5[-1]][0][0] - 0.001, botm_new[Yd4[-1]][0][0]+0.001, 'wwell', 'thiem', 0, 0, 0, 0, 0.0625],
                [wwell_i, wwell_j, botm_new[Yd3[-1]][0][0] - 0.001, botm_new[Yd2[-1]][0][0]+0.001, 'wwell', 'thiem', 0, 0, 0, 0, 0.0625]
                ]

    node_data = pd.DataFrame(mnw_data, columns=mnw_columns)
    node_data = node_data.to_records()

    # stress_period_data_dtype_list = fpm.ModflowMnw2.get_empty_stress_period_data().dtype.descr
    spd_columns = ['per', 'wellid', 'qdes']
    spd_data = [[0, 'cwell', -flowrate/3600], #in m/s
                [1, 'cwell', flowrate/3600],
                [2, 'cwell', -flowrate/3600],
                [3, 'cwell', flowrate / 3600],
                [4, 'cwell', -flowrate / 3600],
                [5, 'cwell', flowrate / 3600],
                [0, 'wwell', flowrate/3600],
                [1, 'wwell', -flowrate/3600],
                [2, 'wwell', flowrate / 3600],
                [3, 'wwell', -flowrate / 3600],
                [4, 'wwell', flowrate / 3600],
                [5, 'wwell', -flowrate / 3600]
                ]
    stress_period_data = pd.DataFrame(spd_data, columns=spd_columns)
    pers = stress_period_data.groupby('per')
    stress_period_data = {i: pers.get_group(i).to_records() for i in
                          range(0, len(mf.dis.nstp.array))}

    mnwmax = 2
    fpm.ModflowMnw2(model=mf,
                    mnwmax=mnwmax,
                    node_data=node_data,
                    stress_period_data=stress_period_data,
                    itmp=[2, 2, 2, 2, 2, 2],  # [mnwmax]*len(mf.dis.nstp.array) #if well inactive in stress period: itmp should be 0 for that sp
                    mnwprnt=2  # level of detail in lst file
                    )

    fpm.ModflowMnwi(model=mf,
                    wel1flag=61,  # unit number to which save .well_out
                    mnwobs=2,
                    wellid_unit_qndflag_qhbflag_concflag=[['wwell',45,0,0],['cwell',46,0,0]],
                    )
# Rch
    nrchop = mf.rch.nrchop
    ipakcb = mf.rch.ipakcb
    rech = mf.rch.rech.array[0][0][0][0]
    irch = mf.rch.irch

    fpm.ModflowRch(
        mf,
        nrchop=nrchop,
        ipakcb=ipakcb,
        rech=rech,
        irch=irch,
        extension='rch',
        unitnumber=unitnumber,
        filenames=None)

# Hob,Bas6 package are written correctly !!

    # write and run flow model
    mf.write_input()
    mf.run_model(silent=False, report=True)

# Btn

    times = np.cumsum(perlen)
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
    ttsmult = tsmult[0]  # The multiplier for successive transport steps within a flow time-step
    ttsmax = mt.btn.ttsmax.array[0]
    extension = "btn"
    unitnumber = None
    filenames = None

    prsity_new = prsity

    for lay in aqf_layers:
        prsity_new[lay] = NewGridValues(nrow,ncol,por_Eaqf)

    for lay in aqt_layers:
        prsity_new[lay] = NewGridValues(nrow, ncol, por_Eaqt)

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
    crch = None
    cevt = None
    mxss = None  # If problem, set to 10000
    dtype = None
    extension = "ssm"
    unitnumber = None
    filenames = None

    ssm_data = {}
    itype = fpt.Mt3dSsm.itype_dict()

    # ssm for boundaries
    temp_initial = 13.8
    for stp in range(nper):
        ssm_data[stp] = []
        for k in range(nlay):
            #impose T top boundary
            if k == 0:
                for i in range(nrow):
                    for j in range(ncol):
                        ssm_data[stp].append((k, i, j, temp_initial, itype['CC']))#alle rijen en kolommen zomer T

            for i in range(nrow):
                if i in NSB:
                    for j in range(ncol):
                        ssm_data[stp].append((k, i, j, temp_initial, itype['CC']))
                        ssm_data[stp].append((k, i, j, temp_initial, itype['CHD']))
                else:  # dit zijn dan oost en west boundaries
                    ssm_data[stp].append((k, i, 0, temp_initial, itype['CC']))
                    ssm_data[stp].append((k, i, 0, temp_initial, itype['WEL']))
                    ssm_data[stp].append((k, i, ncol - 1, temp_initial, itype['CC']))
                    ssm_data[stp].append((k, i, ncol - 1, temp_initial, itype['WEL']))

    # ssm for wells
    warm = temp_initial+5
    cold = temp_initial-5

    # wells
    # get the well locations (lay, row, col) in the zero-based grid (>< lambert coordinates) !condition: all wells already active in first stress period!
    ATES_layers = [Yd6,Yd4,Yd2]
    wwell = []
    cwell = []
    for Yd in ATES_layers:
        for lay in Yd:
            wwell.append((lay,wwell_i,wwell_j))
            cwell.append((lay,cwell_i,cwell_j))

    #     #subdivisions in each aqf layer
    #     !Dis_sub = [5,2,10,6,4,13]

    # # make dictionary with injection rate for every layer for every stress period
    #     inj_rate_perlay = {}
    #     for stp in range(nper):
    #         inj_rate_perlay[stp]=[]
    #         for Yd in range(len(ATES_layers)):
    #             for lay in range(len(ATES_layers[Yd])):
    #                 inj_rate_perlay[stp].append(Q_inj[Yd])
    warm_stp = [0,2,4]
    for stp in range(nper):
        if stp in warm_stp:  # even stress period; warm wells start injecting
            for well in range(len(wwell)):
                ssm_data[stp].append((wwell[well][0], wwell[well][1], wwell[well][2], warm, itype['CC']))
                ssm_data[stp].append((wwell[well][0], wwell[well][1], wwell[well][2], warm, itype['WEL']))
        else:
            for well in range(len(cwell)):
                ssm_data[stp].append((cwell[well][0], cwell[well][1], cwell[well][2], cold, itype['CC']))
                ssm_data[stp].append((cwell[well][0], cwell[well][1], cwell[well][2], cold, itype['WEL']))

    #ssm for top boundary, koppeling met recharge
    crch={}
    for stp in range(nper):
        crch[stp] = []
        crch[stp].append(temp_initial * np.ones((nrow,ncol),dtype=float))

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

    k0_aqt = 0.58 * por_Taqt + 2 * (1 - por_Taqt)
    mol_diff_aqt = k0_aqt / (por_Taqt * 1000 * 4183)

    dmcoef_new = np.zeros((nlay,1))
    dmcoef = mt.dsp.dmcoef[0].array

    for lay in range(nlay):
        if lay in aqf_layers:
            dmcoef_new[lay][0] = mol_diff_aqf

        elif lay in aqt_layers:
            dmcoef_new[lay][0] = mol_diff_aqt

        else:
            dmcoef_new[lay][0] = dmcoef[lay]

    al = mt.dsp.al.array
    al_new = al
    for lay in aqf_layers:
        al_new[lay] = NewGridValues(nrow, ncol, longitudinal)
    for lay in aqt_layers:
        al_new[lay] = NewGridValues(nrow, ncol, longitudinal)

    trpt = mt.dsp.trpt.array #ratio hor/long disperion: typically 0.1, do not change
    trpv = mt.dsp.trpv.array #ratio vert/long dispersion: typically 0.01 do not change
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
    rhob = mt.rct.rhob.array  # bulk density = density solid * (1-total porosity)
    rhob_new = rhob

    rhob_aqf = 2640 * (1 - por_Taqf)  # bulk density = density solid * (1-total porosity)
    rhob_aqt = 2640 * (1 - por_Taqt)

    for lay in aqf_layers:
        rhob_new[lay]=NewGridValues(nrow, ncol, rhob_aqf)
    for lay in aqt_layers:
        rhob_new[lay]=NewGridValues(nrow,ncol,rhob_aqt)

    therm_distr_aqf = 730 / (4183 * 1000)  # thermal distribution coefficient = spec heat capacity solid/(that of water*density water)
    therm_distr_aqt = 1381 / (4183 * 1000)

    extension = "rct"
    isothm = mt.rct.isothm  # linear
    ireact = mt.rct.ireact
    igetsc = mt.rct.igetsc
    prsity2 = None
    srconc = None
    sp1 = mt.rct.sp1[0].array  # thermal distribution coefficient
    sp2 = mt.rct.sp2[0].array #read out but not used when isothm is 1
    rc1 = None
    rc2 = None

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
    mixelm = -1  # -1: TVD, 1: MOC
    percel = 0.5 #mt.adv.percel courant number
    nadvfd = mt.adv.nadvfd
    npmin = mt.adv.npmin
    npmax = mt.adv.npmax
    mxpart = mt.adv.mxpart
    extension = "adv"
    unitnumber = 2
    filenames = None

    flopy.mt3d.Mt3dAdv(
        model=mt,
        mixelm=mixelm,
        percel=percel,
        npmin=npmin,
        npmax=npmax,
        mxpart=mxpart,
        nadvfd=nadvfd,
        extension=extension,
        unitnumber=unitnumber,
        filenames=filenames,
    )

    # write and run transport model
    mt.write_input(check=True)
    mt.run_model(report=True)

    # save deltaT for each run
    os.rename(os.path.join(results_dir, 'Campus_MNW_60_temperature.ucn'),
              os.path.join(results_dir, 'UCN_{}.ucn'.format(run_name)))
    os.remove(os.path.join(results_dir, 'Campus_MNW_60_temperature_S.ucn'))
    os.rename(os.path.join(results_dir, 'Campus_MNW_60.bhd'), os.path.join(results_dir, 'BHD_{}.bhd'.format(run_name)))

# mto to csv
    T = fpt.mt.Mt3dms.load_obs(os.path.join(results_dir, 'Campus_MNW_60_temperature.mto'))

    column_names = ['Step', 'Time(s)']
    nlay_well = len(aqf_layers)
    for lay in range(nlay_well):
        column_names.append('c{}'.format(lay))
        column_names.append('w{}'.format(lay))
    T = pd.DataFrame(T)
    T.columns = column_names
    T['Time (d)'] = T['Time(s)'] / 86400

    # mean Tdifference
    T['MeanTw'] = T.filter(regex='^w').sum(axis=1) / nlay_well
    T['MeanTc'] = T.filter(regex='^c').sum(axis=1) / nlay_well

    T['DeltaT'] = T['MeanTw'] - T['MeanTc']
    T['DeltaT_cold'] = temp_initial - T['MeanTc']
    T['DeltaT_warm'] = T['MeanTw'] - temp_initial

    mto_file =  'MTO_{}.csv'.format(run_name)
    T.to_csv(os.path.join(results_dir, mto_file))

    os.remove(os.path.join(results_dir, 'Campus_MNW_60_temperature.mto'))

    # mnwi_out to csv (to my knowledge there is no flopy function to load this kind of file)
    # separate file created for each MNW object
    # output is average head of observation layers at specified observation location taking into account the well radius

    H_cold = pd.read_fwf(os.path.join(results_dir, 'Campus_MNW_60.0046.mnwobs'), colspecs='infer', #unitnumber observation in filename
                    skiprows=[1])
    H_cold = H_cold.rename(columns={'hwell': 'h_cwell'})
    H_cold = H_cold.set_index('TOTIM')

    H_warm = pd.read_fwf(os.path.join(results_dir, 'Campus_MNW_60.0045.mnwobs'), colspecs='infer',
                         skiprows=[1])
    H_warm = H_warm.rename(columns={'hwell': 'h_wwell'})
    H_warm = H_warm.set_index('TOTIM')

    H_new = pd.concat([H_cold,H_warm], axis=1)  # want one column for each well
    hob_file = 'HOB_{}.csv'.format(run_name)
    H_new.to_csv(os.path.join(results_dir, hob_file))

    return mf, mt