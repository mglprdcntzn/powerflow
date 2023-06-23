import numpy as np
import time
from scipy.linalg import expm
########################################################
def SC(barY, barY0, barS, V0, itmax, prec):
  N = V0.shape[0]
  unos = np.ones((N, 1))
  invY = np.linalg.inv(barY)
  Y0unos = barY0@unos
  conbarS = np.conj(barS)
  ##
  start_time = time.time()
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
  
    ve  = V@unos
    conVinv = np.conj(1/ve)
  
    ve = invY@(conbarS@conVinv -  Y0unos)
  
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRI(barY, barY0, barS, V0, itmax, prec):
  N = V0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
  ##
  start_time = time.time()
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
    
    ve  = V@unos
    Vinv = np.diag(np.squeeze(1/ve))
    h = Vinv@g
      
    AAA = barS@np.square(Vinv)
    FFF = invconY@AAA
    HHH = np.linalg.inv(Id -FFF@np.conj(FFF))@invconY
    Hh  = HHH@h
          
    ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
      
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRI0(barY, barY0, barS, V0, itmax, prec):
  N = V0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
  
  Vinv = np.linalg.inv(V0)
  AAA = barS@np.square(Vinv)
  FFF = invconY@AAA
  XXX = np.linalg.inv(Id -FFF@np.conj(FFF))
  HHH = XXX@invconY
  ##
  start_time = time.time()
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
    
    ve  = V@unos
    Vinv = np.diag(np.squeeze(1/ve))
    h = Vinv@g
      
    Hh  = HHH@h
          
    ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
      
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRIsum(barY, barY0, barS, V0, itmax, prec,M):
  N = V0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
  ##
  start_time = time.time()
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
    
    ve  = V@unos
    Vinv = np.diag(np.squeeze(1/ve))
    h = Vinv@g
      
    AAA = barS@np.square(Vinv)
    FFF = invconY@AAA
    PPP = FFF@np.conj(FFF)
    
    XXX = Id
    for m in range(M):
        XXX = XXX + np.linalg.matrix_power(PPP, m+1)
    
    #XXX = Id + PPP
    
    HHH = XXX@invconY
    Hh  = HHH@h
          
    ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
      
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRIM(barY, barY0, barS, V0, itmax, prec,M):
  N = V0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
  
  Vinv = np.linalg.inv(V0)
  AAA = barS@np.square(Vinv)
  FFF = invconY@AAA
  XXX = np.linalg.inv(Id -FFF@np.conj(FFF))
  ##
  start_time = time.time()
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
    
    ve  = V@unos
    Vinv = np.diag(np.squeeze(1/ve))
    h = Vinv@g
    
    FFF = invconY@barS@np.square(Vinv)
    Hh  = XXX@invconY@h
          
    ve = ve + np.conj(FFF)@Hh - np.conj(Hh)
    
    for m in range(M):
      XXX = Id + FFF@np.conj(XXX)@np.conj(FFF)
    
    
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def IWA(barY, barY0, barS, V0, itmax, prec):
  N = V0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY0 = np.conj(barY0)
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
    
  Vinv = np.linalg.inv(V0)
  A0 = conY0 + np.diag(np.squeeze(conY @ np.conj(V0) @ unos))
  F0 = invconY@Vinv@A0
  H0 = np.linalg.inv(Id -F0@np.conj(F0))@invconY@Vinv
  g0 = gue(V0, barY, barY0, barS)
  F0con = np.conj(F0)
    
  ##
  start_time = time.time()
  #loop
  Deltave = 0*unos
  V= V0
  ve0 = V0@unos
      
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
      
    DeltaV = np.diag(np.squeeze(Deltave))
    Deltag = g0 + DeltaV@conY@np.conj(DeltaV)@unos
    HDve   = H0@Deltag
      
    Deltave = F0con@HDve - np.conj(HDve)
      
    ve = ve0 + Deltave
      
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRX(barY, barY0, barS, V0, itmax, prec):
  N = V0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY0 = np.conj(barY0)
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
  ##
  start_time = time.time()
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
    
    vvv  = V@unos
    Vinv = np.diag(np.squeeze(1/vvv))
     
    AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V) @ unos))
    FFF = invconY@Vinv@AAA
    HHH = np.linalg.inv(Id -FFF@np.conj(FFF))@invconY@Vinv
    Hg  = HHH@g
          
    ve = V@unos
    ve = ve + np.conj(FFF)@Hg - np.conj(Hg)
     
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRX0(barY, barY0, barS, V0, itmax, prec):
  N = V0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY0 = np.conj(barY0)
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
  
  Vinv = np.linalg.inv(V0)
  AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V0) @ unos))
  FFF = invconY@Vinv@AAA
  XXX = np.linalg.inv(Id -FFF@np.conj(FFF))
  ##
  start_time = time.time()
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
    
    vvv  = V@unos
    Vinv = np.diag(np.squeeze(1/vvv))
     
    HHH = XXX@invconY@Vinv
    Hg  = HHH@g
          
    ve = V@unos
    ve = ve + np.conj(FFF)@Hg - np.conj(Hg)
     
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def GS(barY, barY0, barS, V0, itmax, prec):
  N = V0.shape[0]
  unos = np.ones((N, 1))
  ##
  start_time = time.time()
  #Matrix Linv
  L = np.tril(barY)
  U = barY - L
  invL = np.linalg.inv(L)
  #other matrices 
  barY0vec = barY0@unos;
  Svec     = barS@unos;
  cSvec    = np.conj(Svec);
    
  #loop
  V = V0
  ve = V@unos
  permiso = True
  itcont = 0
  while permiso:
    g = gue(V, barY, barY0, barS)
    
    ve = V@unos
    ve = invL@(np.diag(np.squeeze(np.reciprocal(np.conj(ve))))@cSvec  - barY0vec - U@ve)
    
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
  
    V  = np.diag(np.squeeze(ve))
  end_time = time.time()
  return  np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRC(barY, barY0, barS, R0, Phi0, itmax, prec):
  N = R0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY0 = np.conj(barY0)
  conY = np.conj(barY)
    
  permiso = True
  itcont = 0
    
  R = R0
  Phi = Phi0 
  rp = np.block([[R @ unos], [Phi @ unos]])
    
  start_time = time.time()
  while permiso:
    EEE = expm(1j * Phi)
    V = R @ EEE
    g = gue(V, barY, barY0, barS)
  
    AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V) @ unos))
    BBB = V @ conY
    
    Mtx1 = np.block([[Id, Id], [-1j * Id, 1j * Id]])
    Mtx2 = np.block([[AAA, BBB], [np.conj(BBB), np.conj(AAA)]])
    Mtx3 = np.block([[EEE, 1j * V], [np.conj(EEE), -1j * np.conj(V)]])
    
    Mtx = np.real(0.5 * Mtx1 @ Mtx2 @ Mtx3)
    Mtx = np.linalg.inv(Mtx)
    
    Delta = Mtx @ np.block([[np.real(g)], [np.imag(g)]])
    
    rp = rp - Delta
   
    R = np.diag(np.squeeze(rp[:N]))
    Phi = np.diag(np.squeeze(rp[N:]))
    
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
  end_time = time.time()
  ve = R @ expm(1j * Phi) @ unos
    
  return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRCinv(barY, barY0, barS, R0, Phi0, itmax, prec):
  N = R0.shape[0]
  Id = np.eye(N)
  unos = np.ones((N, 1))
  conY0 = np.conj(barY0)
  conY = np.conj(barY)
  invconY = np.linalg.inv(conY)
    
  permiso = True
  itcont = 0
   
  R = R0
  Phi = Phi0 
  rp = np.block([[R @ unos], [Phi @ unos]])
  
  start_time = time.time()
  while permiso:
    EEE = expm(1j * Phi)
    V = R @ EEE
    g = gue(V, barY, barY0, barS)
  
    vvv  = V@unos
    Vinv = np.diag(np.squeeze(1/vvv))
  
    AAA = conY0 + np.diag(np.squeeze(conY @ np.conj(V) @ unos))
    FFF = invconY@Vinv@AAA
    HHH = np.linalg.inv(Id -FFF@np.conj(FFF))@invconY@Vinv
    GGG = -np.conj(FFF)@HHH
    
    Mtx1 = np.block([[np.conj(EEE), EEE], [-1j * Vinv, 1j * np.conj(Vinv)]])
    Mtx2 = np.block([[GGG, np.conj(HHH)], [HHH, np.conj(GGG)]])
    Mtx3 = np.block([[Id, 1j*Id], [Id, -1j * Id]])
    
    Mtx = np.real(0.5 * Mtx1 @ Mtx2 @ Mtx3)
    
    Delta = Mtx @ np.block([[np.real(g)], [np.imag(g)]])
    
    rp = rp - Delta
    
    R = np.diag(np.squeeze(rp[:N]))
    Phi = np.diag(np.squeeze(rp[N:]))
    
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
  end_time = time.time()
  ve = R @ expm(1j * Phi) @ unos

  return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def NRD(barY, barY0, barS, R0, Phi0, itmax, prec):
  N = R0.shape[0]
  unos = np.ones((N, 1))
  invY = np.imag(np.linalg.inv(barY))
    
  permiso = True
  itcont = 0
    
  R  = R0
  Phi = Phi0
  r  = R@unos
  ph = Phi@unos
    
  start_time = time.time()
  while permiso:
    V = R @ expm(1j * Phi)
    g = gue(V, barY, barY0, barS)
    
    r  = r - invY@np.imag(g)
    ph  = ph - invY@np.real(g)
      
    R = np.diag(np.squeeze(r))
    Phi = np.diag(np.squeeze(ph))
      
    itcont = itcont + 1
    if itcont >= itmax or np.linalg.norm(g, ord=2) < prec:
      permiso = 0
      break
    
  end_time = time.time()
  ve = R @ expm(1j * Phi) @ unos
    
  return np.squeeze(ve), itcont, np.round(1000 * (end_time - start_time), 0),np.linalg.norm(g, ord=2)
########################################################
def gue(VVV, barY, barY0, barS):
  unos = np.ones((VVV.shape[0], 1))
  g = VVV @ np.conj(barY0) @ unos + VVV @ np.conj(barY) @ np.conj(VVV @ unos) - barS @ unos
  return g