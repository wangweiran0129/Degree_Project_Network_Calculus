# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: attack.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x61ttack.proto\x12\x06netcal\"Q\n\x07Network\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x1e\n\x06server\x18\x02 \x03(\x0b\x32\x0e.netcal.Server\x12\x1a\n\x04\x66low\x18\x03 \x03(\x0b\x32\x0c.netcal.Flow\"3\n\x06Server\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0c\n\x04rate\x18\x02 \x01(\x01\x12\x0f\n\x07latency\x18\x03 \x01(\x01\"\xc3\x01\n\x04\x46low\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0c\n\x04rate\x18\x02 \x01(\x01\x12\r\n\x05\x62urst\x18\x03 \x01(\x01\x12\x0c\n\x04path\x18\x04 \x03(\x05\x12\x1c\n\x04pmoo\x18\x05 \x01(\x0b\x32\x0e.netcal.Result\x12 \n\x06pmoofp\x18\x06 \x01(\x0b\x32\x10.netcal.FPResult\x12\x1f\n\x07\x64\x65\x62orah\x18\x07 \x01(\x0b\x32\x0e.netcal.Result\x12#\n\tdeborahfp\x18\x08 \x01(\x0b\x32\x10.netcal.FPResult\"\x1d\n\x06Result\x12\x13\n\x0b\x64\x65lay_bound\x18\x01 \x01(\x01\"N\n\x08\x46PResult\x12\x13\n\x0b\x64\x65lay_bound\x18\x01 \x01(\x01\x12-\n\x0eprolonged_flow\x18\x02 \x03(\x0b\x32\x15.netcal.FPCombination\"\x93\x01\n\rFPCombination\x12H\n\x12\x66lows_prolongation\x18\x02 \x03(\x0b\x32,.netcal.FPCombination.FlowsProlongationEntry\x1a\x38\n\x16\x46lowsProlongationEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x62\x06proto3')



_NETWORK = DESCRIPTOR.message_types_by_name['Network']
_SERVER = DESCRIPTOR.message_types_by_name['Server']
_FLOW = DESCRIPTOR.message_types_by_name['Flow']
_RESULT = DESCRIPTOR.message_types_by_name['Result']
_FPRESULT = DESCRIPTOR.message_types_by_name['FPResult']
_FPCOMBINATION = DESCRIPTOR.message_types_by_name['FPCombination']
_FPCOMBINATION_FLOWSPROLONGATIONENTRY = _FPCOMBINATION.nested_types_by_name['FlowsProlongationEntry']
Network = _reflection.GeneratedProtocolMessageType('Network', (_message.Message,), {
  'DESCRIPTOR' : _NETWORK,
  '__module__' : 'attack_pb2'
  # @@protoc_insertion_point(class_scope:netcal.Network)
  })
_sym_db.RegisterMessage(Network)

Server = _reflection.GeneratedProtocolMessageType('Server', (_message.Message,), {
  'DESCRIPTOR' : _SERVER,
  '__module__' : 'attack_pb2'
  # @@protoc_insertion_point(class_scope:netcal.Server)
  })
_sym_db.RegisterMessage(Server)

Flow = _reflection.GeneratedProtocolMessageType('Flow', (_message.Message,), {
  'DESCRIPTOR' : _FLOW,
  '__module__' : 'attack_pb2'
  # @@protoc_insertion_point(class_scope:netcal.Flow)
  })
_sym_db.RegisterMessage(Flow)

Result = _reflection.GeneratedProtocolMessageType('Result', (_message.Message,), {
  'DESCRIPTOR' : _RESULT,
  '__module__' : 'attack_pb2'
  # @@protoc_insertion_point(class_scope:netcal.Result)
  })
_sym_db.RegisterMessage(Result)

FPResult = _reflection.GeneratedProtocolMessageType('FPResult', (_message.Message,), {
  'DESCRIPTOR' : _FPRESULT,
  '__module__' : 'attack_pb2'
  # @@protoc_insertion_point(class_scope:netcal.FPResult)
  })
_sym_db.RegisterMessage(FPResult)

FPCombination = _reflection.GeneratedProtocolMessageType('FPCombination', (_message.Message,), {

  'FlowsProlongationEntry' : _reflection.GeneratedProtocolMessageType('FlowsProlongationEntry', (_message.Message,), {
    'DESCRIPTOR' : _FPCOMBINATION_FLOWSPROLONGATIONENTRY,
    '__module__' : 'attack_pb2'
    # @@protoc_insertion_point(class_scope:netcal.FPCombination.FlowsProlongationEntry)
    })
  ,
  'DESCRIPTOR' : _FPCOMBINATION,
  '__module__' : 'attack_pb2'
  # @@protoc_insertion_point(class_scope:netcal.FPCombination)
  })
_sym_db.RegisterMessage(FPCombination)
_sym_db.RegisterMessage(FPCombination.FlowsProlongationEntry)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FPCOMBINATION_FLOWSPROLONGATIONENTRY._options = None
  _FPCOMBINATION_FLOWSPROLONGATIONENTRY._serialized_options = b'8\001'
  _NETWORK._serialized_start=24
  _NETWORK._serialized_end=105
  _SERVER._serialized_start=107
  _SERVER._serialized_end=158
  _FLOW._serialized_start=161
  _FLOW._serialized_end=356
  _RESULT._serialized_start=358
  _RESULT._serialized_end=387
  _FPRESULT._serialized_start=389
  _FPRESULT._serialized_end=467
  _FPCOMBINATION._serialized_start=470
  _FPCOMBINATION._serialized_end=617
  _FPCOMBINATION_FLOWSPROLONGATIONENTRY._serialized_start=561
  _FPCOMBINATION_FLOWSPROLONGATIONENTRY._serialized_end=617
# @@protoc_insertion_point(module_scope)
