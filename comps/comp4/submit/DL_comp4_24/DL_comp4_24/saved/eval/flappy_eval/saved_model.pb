��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02unknown8��
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
�
action_outputs/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameaction_outputs/kernel
�
)action_outputs/kernel/Read/ReadVariableOpReadVariableOpaction_outputs/kernel*
_output_shapes
:	�*
dtype0
~
action_outputs/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameaction_outputs/bias
w
'action_outputs/bias/Read/ReadVariableOpReadVariableOpaction_outputs/bias*
_output_shapes
:*
dtype0
�
value_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_namevalue_output/kernel
|
'value_output/kernel/Read/ReadVariableOpReadVariableOpvalue_output/kernel*
_output_shapes
:	�*
dtype0
z
value_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namevalue_output/bias
s
%value_output/bias/Read/ReadVariableOpReadVariableOpvalue_output/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*

0
1
2
3
4
5
 
*

0
1
2
3
4
5
�
trainable_variables

layers
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
layer_metrics
 metrics
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
�

!layers
"non_trainable_variables
trainable_variables
#layer_regularization_losses
regularization_losses
	variables
$layer_metrics
%metrics
a_
VARIABLE_VALUEaction_outputs/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEaction_outputs/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

&layers
'non_trainable_variables
trainable_variables
(layer_regularization_losses
regularization_losses
	variables
)layer_metrics
*metrics
_]
VARIABLE_VALUEvalue_output/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEvalue_output/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

+layers
,non_trainable_variables
trainable_variables
-layer_regularization_losses
regularization_losses
	variables
.layer_metrics
/metrics

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
y
serving_default_inputsPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsdense/kernel
dense/biasvalue_output/kernelvalue_output/biasaction_outputs/kernelaction_outputs/bias*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference_signature_wrapper_447
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp)action_outputs/kernel/Read/ReadVariableOp'action_outputs/bias/Read/ReadVariableOp'value_output/kernel/Read/ReadVariableOp%value_output/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*%
f R
__inference__traced_save_552
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasaction_outputs/kernelaction_outputs/biasvalue_output/kernelvalue_output/bias*
Tin
	2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_restore_582��
�
�
>__inference_dense_layer_call_and_return_conditional_losses_256

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
,__inference_action_outputs_layer_call_fn_487

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_action_outputs_layer_call_and_return_conditional_losses_3092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_action_outputs_layer_call_and_return_conditional_losses_478

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_action_outputs_layer_call_and_return_conditional_losses_309

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
x
#__inference_dense_layer_call_fn_467

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�

�
!__inference_signature_wrapper_447

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__wrapped_model_2412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�#
�
__inference__traced_save_552
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop4
0savev2_action_outputs_kernel_read_readvariableop2
.savev2_action_outputs_bias_read_readvariableop2
.savev2_value_output_kernel_read_readvariableop0
,savev2_value_output_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9bfaa3cbf36d434a8d3990d29d3f304e/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop0savev2_action_outputs_kernel_read_readvariableop.savev2_action_outputs_bias_read_readvariableop.savev2_value_output_kernel_read_readvariableop,savev2_value_output_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*K
_input_shapes:
8: :	�:�:	�::	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�
�
>__inference_model_layer_call_and_return_conditional_losses_409

inputs
	dense_392
	dense_394
value_output_397
value_output_399
action_outputs_402
action_outputs_404
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs	dense_392	dense_394*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2562
dense/StatefulPartitionedCall�
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_397value_output_399*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_value_output_layer_call_and_return_conditional_losses_2822&
$value_output/StatefulPartitionedCall�
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_outputs_402action_outputs_404*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_action_outputs_layer_call_and_return_conditional_losses_3092(
&action_outputs/StatefulPartitionedCall�
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

�
#__inference_model_layer_call_fn_387

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_3702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_value_output_layer_call_and_return_conditional_losses_497

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_dense_layer_call_and_return_conditional_losses_458

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_model_layer_call_and_return_conditional_losses_370

inputs
	dense_353
	dense_355
value_output_358
value_output_360
action_outputs_363
action_outputs_365
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs	dense_353	dense_355*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2562
dense/StatefulPartitionedCall�
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_358value_output_360*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_value_output_layer_call_and_return_conditional_losses_2822&
$value_output/StatefulPartitionedCall�
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_outputs_363action_outputs_365*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_action_outputs_layer_call_and_return_conditional_losses_3092(
&action_outputs/StatefulPartitionedCall�
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_value_output_layer_call_and_return_conditional_losses_282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������:::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_model_layer_call_and_return_conditional_losses_327

inputs
	dense_267
	dense_269
value_output_293
value_output_295
action_outputs_320
action_outputs_322
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs	dense_267	dense_269*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2562
dense/StatefulPartitionedCall�
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_293value_output_295*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_value_output_layer_call_and_return_conditional_losses_2822&
$value_output/StatefulPartitionedCall�
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_outputs_320action_outputs_322*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_action_outputs_layer_call_and_return_conditional_losses_3092(
&action_outputs/StatefulPartitionedCall�
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
>__inference_model_layer_call_and_return_conditional_losses_347

inputs
	dense_330
	dense_332
value_output_335
value_output_337
action_outputs_340
action_outputs_342
identity

identity_1��&action_outputs/StatefulPartitionedCall�dense/StatefulPartitionedCall�$value_output/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputs	dense_330	dense_332*
Tin
2*
Tout
2*(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_2562
dense/StatefulPartitionedCall�
$value_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0value_output_335value_output_337*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_value_output_layer_call_and_return_conditional_losses_2822&
$value_output/StatefulPartitionedCall�
&action_outputs/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0action_outputs_340action_outputs_342*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_action_outputs_layer_call_and_return_conditional_losses_3092(
&action_outputs/StatefulPartitionedCall�
IdentityIdentity/action_outputs/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity-value_output/StatefulPartitionedCall:output:0'^action_outputs/StatefulPartitionedCall^dense/StatefulPartitionedCall%^value_output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::2P
&action_outputs/StatefulPartitionedCall&action_outputs/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$value_output/StatefulPartitionedCall$value_output/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�

*__inference_value_output_layer_call_fn_506

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_value_output_layer_call_and_return_conditional_losses_2822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�

�
#__inference_model_layer_call_fn_426

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*:
_output_shapes(
&:���������:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU2*0J 8*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_4092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
__inference__wrapped_model_241

inputs.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource5
1model_value_output_matmul_readvariableop_resource6
2model_value_output_biasadd_readvariableop_resource7
3model_action_outputs_matmul_readvariableop_resource8
4model_action_outputs_biasadd_readvariableop_resource
identity

identity_1��
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulinputs)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model/dense/BiasAdd}
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model/dense/Relu�
(model/value_output/MatMul/ReadVariableOpReadVariableOp1model_value_output_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02*
(model/value_output/MatMul/ReadVariableOp�
model/value_output/MatMulMatMulmodel/dense/Relu:activations:00model/value_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/value_output/MatMul�
)model/value_output/BiasAdd/ReadVariableOpReadVariableOp2model_value_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model/value_output/BiasAdd/ReadVariableOp�
model/value_output/BiasAddBiasAdd#model/value_output/MatMul:product:01model/value_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/value_output/BiasAdd�
*model/action_outputs/MatMul/ReadVariableOpReadVariableOp3model_action_outputs_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*model/action_outputs/MatMul/ReadVariableOp�
model/action_outputs/MatMulMatMulmodel/dense/Relu:activations:02model/action_outputs/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/action_outputs/MatMul�
+model/action_outputs/BiasAdd/ReadVariableOpReadVariableOp4model_action_outputs_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model/action_outputs/BiasAdd/ReadVariableOp�
model/action_outputs/BiasAddBiasAdd%model/action_outputs/MatMul:product:03model/action_outputs/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/action_outputs/BiasAdd�
model/action_outputs/SoftmaxSoftmax%model/action_outputs/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model/action_outputs/Softmaxz
IdentityIdentity&model/action_outputs/Softmax:softmax:0*
T0*'
_output_shapes
:���������2

Identity{

Identity_1Identity#model/value_output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:���������:::::::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�"
�
__inference__traced_restore_582
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias,
(assignvariableop_2_action_outputs_kernel*
&assignvariableop_3_action_outputs_bias*
&assignvariableop_4_value_output_kernel(
$assignvariableop_5_value_output_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_action_outputs_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_action_outputs_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_value_output_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_value_output_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
inputs/
serving_default_inputs:0���������B
action_outputs0
StatefulPartitionedCall:0���������@
value_output0
StatefulPartitionedCall:1���������tensorflow/serving/predict:�s
�"
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
0__call__
*1&call_and_return_all_conditional_losses
2_default_save_signature"�
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputs"}, "name": "inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_outputs", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_outputs", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["inputs", 0, 0]], "output_layers": [["action_outputs", 0, 0], ["value_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputs"}, "name": "inputs", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["inputs", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_outputs", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_outputs", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "value_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["inputs", 0, 0]], "output_layers": [["action_outputs", 0, 0], ["value_output", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "inputs", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "inputs"}}
�


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*5&call_and_return_all_conditional_losses
6__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "action_outputs", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "action_outputs", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*7&call_and_return_all_conditional_losses
8__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "value_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "value_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
�
trainable_variables

layers
non_trainable_variables
layer_regularization_losses
regularization_losses
	variables
layer_metrics
 metrics
0__call__
2_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
:	�2dense/kernel
:�2
dense/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
�

!layers
"non_trainable_variables
trainable_variables
#layer_regularization_losses
regularization_losses
	variables
$layer_metrics
%metrics
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
(:&	�2action_outputs/kernel
!:2action_outputs/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

&layers
'non_trainable_variables
trainable_variables
(layer_regularization_losses
regularization_losses
	variables
)layer_metrics
*metrics
6__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
&:$	�2value_output/kernel
:2value_output/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

+layers
,non_trainable_variables
trainable_variables
-layer_regularization_losses
regularization_losses
	variables
.layer_metrics
/metrics
8__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�2�
#__inference_model_layer_call_fn_426
#__inference_model_layer_call_fn_387�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
>__inference_model_layer_call_and_return_conditional_losses_347
>__inference_model_layer_call_and_return_conditional_losses_327�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_241�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *%�"
 �
inputs���������
�2�
>__inference_dense_layer_call_and_return_conditional_losses_458�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_dense_layer_call_fn_467�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_action_outputs_layer_call_and_return_conditional_losses_478�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_action_outputs_layer_call_fn_487�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_value_output_layer_call_and_return_conditional_losses_497�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_value_output_layer_call_fn_506�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
/B-
!__inference_signature_wrapper_447inputs�
__inference__wrapped_model_241�
/�,
%�"
 �
inputs���������
� "w�t
:
action_outputs(�%
action_outputs���������
6
value_output&�#
value_output����������
G__inference_action_outputs_layer_call_and_return_conditional_losses_478]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
,__inference_action_outputs_layer_call_fn_487P0�-
&�#
!�
inputs����������
� "�����������
>__inference_dense_layer_call_and_return_conditional_losses_458]
/�,
%�"
 �
inputs���������
� "&�#
�
0����������
� w
#__inference_dense_layer_call_fn_467P
/�,
%�"
 �
inputs���������
� "������������
>__inference_model_layer_call_and_return_conditional_losses_327�
7�4
-�*
 �
inputs���������
p

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
>__inference_model_layer_call_and_return_conditional_losses_347�
7�4
-�*
 �
inputs���������
p 

 
� "K�H
A�>
�
0/0���������
�
0/1���������
� �
#__inference_model_layer_call_fn_387�
7�4
-�*
 �
inputs���������
p

 
� "=�:
�
0���������
�
1����������
#__inference_model_layer_call_fn_426�
7�4
-�*
 �
inputs���������
p 

 
� "=�:
�
0���������
�
1����������
!__inference_signature_wrapper_447�
9�6
� 
/�,
*
inputs �
inputs���������"w�t
:
action_outputs(�%
action_outputs���������
6
value_output&�#
value_output����������
E__inference_value_output_layer_call_and_return_conditional_losses_497]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_value_output_layer_call_fn_506P0�-
&�#
!�
inputs����������
� "����������