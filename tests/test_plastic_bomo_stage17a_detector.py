import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from plastic_bomo_stage17a_detector_common import (load,detector_protocol,EpochSampler,schedule,
    TraceDatasetMixin,headroom,verify_generation,function_with_global)

class NativeSlottedDataset:
    __slots__=('value',)
    def __init__(self,value): self.value=value

def native_factory(value,*,flag=True):
    return NativeSlottedDataset((value,flag))

def native_method(self,value):
    return native_factory(value,flag=False)

class DetectorTests(unittest.TestCase):
    def test_construct_traced_slotted_dataset_without_class_swap(self):
        class Traced(TraceDatasetMixin,NativeSlottedDataset): pass
        factory=function_with_global(native_factory,'NativeSlottedDataset',NativeSlottedDataset,Traced)
        method=function_with_global(native_method,'native_factory',native_factory,factory)
        result=method(None,123)
        self.assertIsInstance(result,Traced)
        self.assertEqual(result.value,(123,False))
        self.assertIs(type(native_factory(123)),NativeSlottedDataset)
        self.assertEqual(factory(123).value,(123,True))
        with self.assertRaises(RuntimeError):
            function_with_global(native_factory,'NativeSlottedDataset',object,Traced)

    def test_protocol_frozen(self):
        cfg=load(ROOT/'configs/plastic_bomo_stage17a.json')
        self.assertEqual(detector_protocol(cfg)['epochs'],150)
        changed=copy.deepcopy(cfg); changed['detector']['mosaic']=0
        with self.assertRaises(RuntimeError): detector_protocol(changed)

    def test_sampler_permutation_and_position_independence(self):
        names=[f'base_{i:03d}' for i in range(138)]+[f'extra_{i:03d}' for i in range(80)]
        sampler=EpochSampler(names); reverse=EpochSampler(list(reversed(names)))
        for epoch in (0,1,139,149):
            sampler.set_epoch(epoch); reverse.set_epoch(epoch)
            delivered=[names[i] for i in sampler]
            other=[reverse.names[i] for i in reverse]
            self.assertEqual(delivered,other); self.assertEqual(delivered,schedule(names,epoch))
            self.assertEqual(len(set(delivered)),218)
        self.assertNotEqual(schedule(names,0),schedule(names,1))

    def test_trace_survives_transforms_and_nested_mosaic_fetch(self):
        class Fake:
            im_files=['base_000.jpg','extra_000.png']
            def get_image_and_label(self,index): return {'image':self.im_files[index]}
            def __getitem__(self,index):
                row=self.get_image_and_label(index); self.get_image_and_label(1)
                return {'formatted':row}
        class Traced(TraceDatasetMixin,Fake): pass
        ds=Traced(); row=ds[0]
        self.assertEqual(row['stage17_fetches'],('base_000','extra_000'))
        self.assertIsNone(ds._stage17_calls)
        self.assertEqual(ds[1]['stage17_fetches'],('extra_000','extra_000'))

    def rows(self,values,classes=None):
        return [{'arm':a,'map50_95':v,'per_class':{c:{'ap50_95':(classes or {}).get(a,v)} for c in ('flash','black')}}
                for a,v in zip(('RR','VA','VB','VC'),values)]

    def test_predeclared_statuses_and_sample_std(self):
        import statistics
        summary=headroom(self.rows([.4,.42,.41,.43]))
        self.assertEqual(summary['status'],'VANILLA_SYNTHETIC_HEADROOM_REPRODUCIBLE')
        self.assertAlmostEqual(summary['vanilla_std_pp'],100*statistics.stdev([.42,.41,.43]))
        self.assertEqual(summary['wins_vs_RR'],3)
        self.assertEqual(headroom(self.rows([.4,.401,.399,.402]))['status'],'SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE')
        self.assertEqual(headroom(self.rows([.4,.42,.41,.43],{'VA':.35,'VB':.35,'VC':.35}))['status'],'SYNTHETIC_HEADROOM_CLASS_UNSTABLE')

    def test_all_actual_downloaded_banks_and_probes(self):
        root=ROOT/'results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability'
        banks=root/'banks_20261007_035832'
        if not banks.is_dir(): self.skipTest('downloaded server images unavailable')
        slots,records=verify_generation(banks,root/'slots_frozen_20261007_020941',root/'pregeneration_20261007_024210/probe')
        self.assertEqual(len(slots),80); self.assertEqual(len(records),240)

if __name__=='__main__': unittest.main()
