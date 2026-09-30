"""Stage12A set-level descriptors and strictly within-family utility audit."""
from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
def save(p,x):Path(p).write_text(json.dumps(clean(x),indent=2,allow_nan=False),encoding="utf-8")
def dist(a,b):return 1-np.clip(a@b.T,-1,1)
def q(v,p):return float(np.quantile(v,p)) if len(v) else None
def rankdata(v):
 a=np.asarray(v,float);order=np.argsort(a,kind="mergesort");r=np.empty(len(a),float);i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]:j+=1
  r[order[i:j]]=(i+j+1)/2;i=j
 return r
def spearman(x,y):
 rx,ry=rankdata(x),rankdata(y);sx,sy=rx.std(),ry.std()
 return float(np.mean((rx-rx.mean())*(ry-ry.mean()))/(sx*sy)) if sx>0 and sy>0 else 0.
def entropy(vals):
 c=np.asarray(list(Counter(vals).values()),float);p=c/c.sum();return float(-(p*np.log(p)).sum()/max(np.log(len(c)),1e-12)) if len(c)>1 else 0.
def ess(d,sigma):
 if len(d)<2:return 1.
 k=np.exp(-(d*d)/(2*max(sigma,1e-9)**2));return float(len(d)**2/(len(d)+2*k[np.triu_indices(len(d),1)].sum()))
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","manifest","embedding_npz","embedding_manifest","utility","family_validity","stage3r","stage11a","output"):p.add_argument("--"+n,type=Path,required=True)
 a=p.parse_args();cfg=load(a.protocol);rows=load(a.manifest)["records"];meta=load(a.embedding_manifest)["records"]
 with np.load(a.embedding_npz) as z:F=z["features"];G=z["gradients"]
 assert len(rows)==len(meta)==len(F)==len(G);bankdoc=load(a.stage3r/"oof_feature_bank_manifest_corrected.json");realF={};realmeta={}
 for e in bankdoc["folds"]:
  f=int(e["fold"])
  with np.load(a.stage3r/f"oof_real_bank_corrected_fold{f}.npz") as z:realF[f]=z["features"].astype(float);realF[f]/=np.linalg.norm(realF[f],axis=1,keepdims=True).clip(1e-12);realmeta[f]=e["records"]
 with np.load(a.stage11a/"real_oof_gradient_embeddings.npz") as z:rG=z["embeddings"].astype(float);rgfold=z["folds"];rgcls=z["class_ids"];rgpair=z["pair_ids"].astype(str);rgid=z["instance_ids"].astype(str)
 # Frozen group references and bandwidths.
 refs={}
 for f in range(3):
  for ci in range(6):
   idx=[i for i,x in enumerate(realmeta[f]) if int(x["class_id"])==ci];X=realF[f][idx];pairs=np.asarray([realmeta[f][i]["pair_id"] for i in idx]);D=dist(X,X);mask=(pairs[:,None]!=pairs[None,:])&(~np.eye(len(X),dtype=bool));vals=D[mask];nn=np.asarray([D[i][(pairs!=pairs[i])].min() for i in range(len(X))]);refs[(f,ci)]={"F":X,"pair":pairs,"id":np.asarray([realmeta[f][i]["instance_id"] for i in idx]),"sigma":float(np.median(vals)),"q95":q(nn,.95),"radius":{realmeta[f][idx[i]]["instance_id"]:float(nn[i]) for i in range(len(idx))}}
 gradrefs={}
 for f in range(3):
  for ci in range(6):
   ii=np.where((rgfold==f)&(rgcls==ci))[0];X=rG[ii];D=dist(X,X);pp=rgpair[ii];vals=D[(pp[:,None]!=pp[None,:])&(~np.eye(len(X),dtype=bool))];gradrefs[(f,ci)]={"G":X,"pair":pp,"sigma":float(np.median(vals))}
 groups=defaultdict(list)
 for i,x in enumerate(rows):groups[(x["family_id"],x["arm"])].append(i)
 align=[];coverage=[];redund=[];composition=[];gradient=[];descriptor={}
 for key,ii in sorted(groups.items()):
  fid,arm=key;az=[];gt=[]
  for i in ii:
   x=rows[i];r=refs[(x["fold"],x["class_id"])];d=dist(F[i:i+1],r["F"])[0];valid=r["pair"]!=x["physical_pair_id"];nearest=float(d[valid].min());rad=max(r["radius"].get(x["source_instance_id"],np.median(list(r["radius"].values()))),1e-12);az.append(nearest/rad);gt.append(nearest>r["q95"])
  cov=[]
  for f in range(3):
   for ci in range(6):
    jj=[i for i in ii if rows[i]["fold"]==f and rows[i]["class_id"]==ci]
    if not jj:continue
    r=refs[(f,ci)];D=dist(r["F"],F[jj])
    for k in range(len(r["F"])):
     allow=np.asarray([rows[j]["physical_pair_id"]!=r["pair"][k] for j in jj]);
     if allow.any():cov.append(float(D[k,allow].min()/max(r["radius"][r["id"][k]],1e-12)))
  fd=[];fe=[];gd=[];ge=[]
  for f in range(3):
   for ci in range(6):
    jj=[i for i in ii if rows[i]["fold"]==f and rows[i]["class_id"]==ci]
    if len(jj)<2:continue
    d=dist(F[jj],F[jj]);vals=d[np.triu_indices(len(jj),1)];fd.extend(vals.tolist());fe.append(ess(d,refs[(f,ci)]["sigma"])/len(jj));dg=dist(G[jj],G[jj]);gd.extend(dg[np.triu_indices(len(jj),1)].tolist());ge.append(ess(dg,gradrefs[(f,ci)]["sigma"])/len(jj))
  pairs=[rows[i]["physical_pair_id"] for i in ii];classes=[rows[i]["class_name"] for i in ii];seeds=[str(rows[i]["generation_seed"]) for i in ii];class_counts=Counter(classes);ideal=1/6;class_dev=float(sum(abs(class_counts.get(c,0)/len(ii)-ideal) for c in CLASSES));gradmis=[]
  for i in ii:
   x=rows[i];r=gradrefs[(x["fold"],x["class_id"])];valid=r["pair"]!=x["physical_pair_id"];gradmis.append(float(dist(G[i:i+1],r["G"])[0][valid].min()))
  A={"family_id":fid,"arm":arm,"n":len(ii),"nearest_real_normalized_distance":{"median":q(az,.5),"q90":q(az,.9)},"fraction_beyond_real_q95":float(np.mean(gt)),"fraction_beyond_2x_real_q95":float(np.mean(np.asarray(az)>2))};B={"family_id":fid,"arm":arm,"real_support_normalized_distance":{"median":q(cov,.5),"q90":q(cov,.9)},"covered_real_count":len(cov)};C={"family_id":fid,"arm":arm,"nearest_neighbor_distance":q(fd,0) if fd else None,"median_pairwise_distance":q(fd,.5),"rbf_effective_sample_size_normalized":float(np.mean(fe)) if fe else None,"rbf_bandwidth":"median different-pair real-real distance per fold/class"};D={"family_id":fid,"arm":arm,"unique_source_count":len(set(x["source_instance_id"] for x in (rows[i] for i in ii))),"unique_pair_count":len(set(pairs)),"source_entropy":entropy([rows[i]["source_instance_id"] for i in ii]),"pair_entropy":entropy(pairs),"class_entropy":entropy(classes),"seed_entropy":entropy(seeds),"class_l1_deviation_from_uniform":class_dev};E={"family_id":fid,"arm":arm,"mean_nearest_real_gradient_misalignment":float(np.mean(gradmis)),"q90_nearest_real_gradient_misalignment":q(gradmis,.9),"gradient_rbf_effective_sample_size_normalized":float(np.mean(ge)) if ge else None,"diagnostic_only":True};align.append(A);coverage.append(B);redund.append(C);composition.append(D);gradient.append(E);descriptor[key]={"support_alignment":-A["nearest_real_normalized_distance"]["q90"],"marginal_real_coverage":-B["real_support_normalized_distance"]["q90"],"feature_ess":C["rbf_effective_sample_size_normalized"],"composition_balance":D["pair_entropy"],"gradient_alignment":-E["mean_nearest_real_gradient_misalignment"]}
 save(a.output/"set_real_support_alignment.json",{"direction":"lower distance is better","records":align});save(a.output/"set_marginal_real_coverage.json",{"direction":"lower uncovered-real distance is better","records":coverage});save(a.output/"set_feature_redundancy.json",{"direction":"higher normalized ESS is less redundant","records":redund});save(a.output/"set_composition_balance.json",{"direction":"higher normalized entropy is more balanced","records":composition});save(a.output/"set_gradient_distribution.json",{"direction":"lower misalignment is better; diagnostic only","records":gradient})
 util=load(a.utility)["records"];U={(x["family_id"],x["arm"]):x["map50_95"] for x in util};role={x["family_id"]:x["role"] for x in load(a.family_validity)["families"]};primary=[f for f,r in role.items() if r=="primary"]
 winner=[];rankrows=[]
 for fid in sorted(set(k[0] for k in descriptor)):
  keys=[k for k in descriptor if k[0]==fid and k in U];
  if len(keys)<2:continue
  uw=max(keys,key=lambda k:U[k]);ur={k:float(rankdata([U[j] for j in keys])[keys.index(k)]) for k in keys}
  for name in next(iter(descriptor.values())):
   valid=[k for k in keys if descriptor[k][name] is not None];dw=max(valid,key=lambda k:descriptor[k][name]);pairs=0;correct=0
   for i in range(len(valid)):
    for j in range(i+1,len(valid)):
     if U[valid[i]]==U[valid[j]] or descriptor[valid[i]][name]==descriptor[valid[j]][name]:continue
     pairs+=1;correct+=((U[valid[i]]-U[valid[j]])*(descriptor[valid[i]][name]-descriptor[valid[j]][name])>0)
   winner.append({"family_id":fid,"family_role":role[fid],"descriptor":name,"descriptor_winner":dw[1],"utility_winner":uw[1],"winner_match":dw==uw,"pairwise_correct":int(correct),"pairwise_total":pairs,"pairwise_consistency":correct/pairs if pairs else None})
   for k in valid:rankrows.append({"family_id":fid,"family_role":role[fid],"arm":k[1],"descriptor":name,"descriptor_value":descriptor[k][name],"utility":U[k],"descriptor_rank_percentile":float((rankdata([descriptor[j][name] for j in valid])[valid.index(k)]-1)/max(1,len(valid)-1)),"utility_rank_percentile":float((rankdata([U[j] for j in valid])[valid.index(k)]-1)/max(1,len(valid)-1))})
 save(a.output/"descriptor_winner_consistency.json",{"records":winner,"within_family_only":True})
 associations=[];rng=np.random.default_rng(cfg["bootstrap_seed"])
 for name in next(iter(descriptor.values())):
  z=[x for x in rankrows if x["descriptor"]==name and x["family_id"] in primary];x=np.asarray([r["descriptor_rank_percentile"] for r in z]);y=np.asarray([r["utility_rank_percentile"] for r in z]);rho=spearman(x,y);lofo={f:spearman([r["descriptor_rank_percentile"] for r in z if r["family_id"]!=f],[r["utility_rank_percentile"] for r in z if r["family_id"]!=f]) for f in primary};wc=[r["pairwise_consistency"] for r in winner if r["descriptor"]==name and r["family_id"] in primary and r["pairwise_consistency"] is not None];boot=[]
  for _ in range(cfg["bootstrap_replicates"]):boot.append(float(np.mean(rng.choice(wc,len(wc),replace=True))))
  associations.append({"descriptor":name,"primary_family_count":len(primary),"winner_consistency":float(np.mean(wc)),"rank_spearman_rho":rho,"leave_one_family_out_rho":lofo,"lofo_sign_stable":all(v>0 for v in lofo.values()),"family_bootstrap_winner_consistency_ci95":[q(boot,.025),q(boot,.975)],"passes":len(primary)>=cfg["minimum_primary_families"] and float(np.mean(wc))>=cfg["winner_consistency_min"] and abs(rho)>=.50 and all(v>0 for v in lofo.values())})
 save(a.output/"descriptor_rank_association.json",{"records":associations,"within_family_rank_percentiles":rankrows,"family_unit_bootstrap":True})
 axes=("support_alignment","marginal_real_coverage","feature_ess","composition_balance");dom=[];pareto=[]
 for fid in sorted(set(k[0] for k in descriptor)):
  keys=[k for k in descriptor if k[0]==fid and k in U];front=[]
  for k in keys:
   dominated=any(all(descriptor[j][d]>=descriptor[k][d] for d in axes) and any(descriptor[j][d]>descriptor[k][d] for d in axes) for j in keys if j!=k);front.append(k[1]) if not dominated else None
  win=max(keys,key=lambda k:U[k]);pareto.append({"family_id":fid,"family_role":role[fid],"utility_winner":win[1],"pareto_front":front,"winner_on_frontier":win[1] in front,"winner_dominated":win[1] not in front})
  for i in range(len(keys)):
   for j in range(len(keys)):
    if i==j:continue
    ka,kb=keys[i],keys[j];dominates=all(descriptor[ka][d]>=descriptor[kb][d] for d in axes) and any(descriptor[ka][d]>descriptor[kb][d] for d in axes)
    if dominates:dom.append({"family_id":fid,"family_role":role[fid],"dominant_arm":ka[1],"dominated_arm":kb[1],"utility_order_correct":U[ka]>U[kb]})
 pd=[x for x in dom if x["family_id"] in primary];pp=[x for x in pareto if x["family_id"] in primary];dom_wc=float(np.mean([x["utility_order_correct"] for x in pd])) if pd else 0.;pareto_gate=float(np.mean([x["winner_on_frontier"] for x in pp]))>=.7
 save(a.output/"set_level_dominance_audit.json",{"axes":axes,"records":dom,"primary_winner_consistency":dom_wc,"passes":len(primary)>=4 and dom_wc>=.7});save(a.output/"set_pareto_utility_audit.json",{"axes":axes,"records":pareto,"primary_winner_on_frontier_fraction":float(np.mean([x["winner_on_frontier"] for x in pp])),"passes":len(primary)>=4 and pareto_gate})
 single=[x["descriptor"] for x in associations if x["passes"] and x["descriptor"]!="gradient_alignment"];multi=(len(primary)>=4 and (dom_wc>=.7 or pareto_gate));status="SET_LEVEL_UTILITY_STRUCTURE_SUPPORTED" if single else "MULTIAXIS_SET_UTILITY_STRUCTURE_SUPPORTED" if multi else "SET_LEVEL_UTILITY_STRUCTURE_NOT_SUPPORTED"
 final={"status":status,"primary_family_count":len(primary),"single_descriptor_support":single,"dominance_support":dom_wc>=.7,"pareto_support":pareto_gate,"STAGE12B_MECHANISM_DESIGN_AUTHORIZED":bool(single or multi),"STAGE12B_AUTOMATICALLY_STARTED":False,"F1_excluded_incomplete":True,"F5_secondary_only":True,"official_validation_new_evaluation_count":0,"synthetic_generation_count":0,"selection_rerun_count":0,"generator_training_count":0,"detector_training_count":0,"cross_family_raw_metric_pooling":False};save(a.output/"stage12a_status.json",final);save(a.output/"plastic_bomo_external_sanity.json",{"status":"PLASTIC_BOMO_EXTERNAL_SANITY_NOT_AVAILABLE","used_for_gate":False})
 (a.output/"STAGE12A_REPORT.md").write_text(f"# DeepPCB Stage12A Historical Set-Level Marginal Utility Audit\n\nStatus: **{status}**\n\nPrimary families: F2, F3, F4, F6. F1 is incomplete; F5 is secondary exploratory only. All associations use within-family ranks; no raw metric is pooled across families.\n\nSupported single descriptors: {', '.join(single) if single else 'none'}. Dominance consistency: {dom_wc:.3f}. Pareto winner-on-frontier fraction: {float(np.mean([x['winner_on_frontier'] for x in pp])):.3f}.\n\nNo generation, training, detector rerun, or official validation access was performed. Gradient geometry is diagnostic and does not define the innovation claim.\n",encoding="utf-8");print(json.dumps(final,indent=2))
if __name__=="__main__":main()
