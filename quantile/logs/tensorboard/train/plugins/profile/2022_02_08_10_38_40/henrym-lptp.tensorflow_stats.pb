"��
BHostIDLE"IDLE1��ngv�@A��ngv�@a2`4{-��?i2`4{-��?�Unknown
wHostExp"&model/likelihood_layer/exp/forward/Exp(1^�I�|@9^�I�|@A^�I�|@I^�I�|@a�c�,�~?i�@ӗ�?�Unknown
�Host	ReverseV2"@model/gp_layer/fill_triangular/forward/fill_triangular/ReverseV2(1��v���{@9��v���{@A��v���{@I��v���{@a����om}?ii��rG�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1sh��|��@9sh��|��@Aq=
ף{@Iq=
ף{@a7�o��|?i��
��?�Unknown
�HostStridedSliceGrad"Egradient_tape/model/likelihood_layer/strided_slice_3/StridedSliceGrad(1�z�G�v@9�z�G�v@A�z�G�v@I�z�G�v@a>pL�p(x?i���i[��?�Unknown
�Host
LogicalAnd"wmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/cond/_126/model/gp_layer/map/while/LogicalAnd(1;�O��Bv@9�)g�]@A;�O��Bv@I�)g�]@a��L.�w?iV+�ū��?�Unknown
kHostMul"model/likelihood_layer/mul(1F���Բr@9F���Բr@AF���Բr@IF���Բr@a�`��4�s?i�/j�?�Unknown
�HostStridedSliceGrad"Egradient_tape/model/likelihood_layer/strided_slice_4/StridedSliceGrad(1��Mb�q@9��Mb�q@A��Mb�q@I��Mb�q@a����s�r?i.� .�?�Unknown
�	HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/sub_3_grad/Sum_1(1j�t�hq@9j�t�ha@Aj�t�hq@Ij�t�ha@a�r?iL�B�
S�?�Unknown
g
HostLess"Adam/gradients/Less_4(1���S�m@9���S�m@A���S�m@I���S�m@a���:nno?i��}�xr�?�Unknown
�HostBatchMatMulV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_1_grad/MatMul(1�Vl@9�V\@A�Vl@I�V\@a�e�y��m?ie���;��?�Unknown
}HostSum",gradient_tape/model/likelihood_layer/add/Sum(1�A`��nj@9�A`��nj@A�A`��nj@I�A`��nj@a���mol?i.\eNS��?�Unknown
�HostBatchMatMulV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_1_grad/MatMul_1(1y�&1Tj@9y�&1TZ@Ay�&1Tj@Iy�&1TZ@a0� w��k?i�|�FN��?�Unknown
�HostMatrixTriangularSolve"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve(1���S�i@9���S�Y@A���S�i@I���S�Y@a}�j`�)k?i��<3x��?�Unknown
�HostCholesky"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Cholesky(1��~j��a@9��~j��Q@A��~j��a@I��~j��Q@a�'�_��b?i� x��?�Unknown
�HostBatchMatMulV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/MatMul_1(1V-��ca@9V-��cQ@AV-��ca@IV-��cQ@a���zb?i��T��?�Unknown
�HostMatrixTriangularSolve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/triangular_solve/MatrixTriangularSolve(1!�rh�E`@9!�rh�EP@A!�rh�E`@I!�rh�EP@aN(�V�Ja?i!��
>�?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1����̸h@9����̸h@A�G�z�^@I�G�z�^@an
��f`?i����*�?�Unknown
}HostMul",gradient_tape/model/likelihood_layer/mul/Mul(1w��/=^@9w��/=^@Aw��/=^@Iw��/=^@aF��x`?iwK�{�:�?�Unknown
mHostAddV2"model/likelihood_layer/add(1R����]@9R����]@AR����]@IR����]@aqs\C�_?i��@��J�?�Unknown
�HostLog"Bmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/Log(1V-���\@9V-���\@AV-���\@IV-���\@a��t��^?i4�m�Y�?�Unknown
�HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/MatMul(1#��~j<[@9#��~j<K@A#��~j<[@I#��~j<K@a� ����\?i)�le[h�?�Unknown
eHostSum"model/gp_layer/Sum_1(1��(\�*[@9��(\�*[@A��(\�*[@I��(\�*[@ad[����\?i�V��v�?�Unknown
�HostSign"Cmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/Sign(1L7�A`�Z@9L7�A`�Z@AL7�A`�Z@IL7�A`�Z@a�ٳ��Y\?i��6����?�Unknown
�HostMatrixTriangularSolve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/triangular_solve/MatrixTriangularSolve(1fffffZ@9fffffJ@AfffffZ@IfffffJ@a��. ��[?i=���ؒ�?�Unknown
�HostSub"Bmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub(1
ףp=�W@9
ףp=�W@A
ףp=�W@I
ףp=�W@ax���aY?i)Aǉ��?�Unknown
�HostBroadcastTo"0gradient_tape/model/likelihood_layer/BroadcastTo(1�S㥛�W@9�S㥛�W@A�S㥛�W@I�S㥛�W@a�),��(Y?i>�<��?�Unknown
�HostMatrixBandPart"{model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/MatrixBandPart(1m����:U@9m����:E@Am����:U@Im����:E@a�(���V?iN.�:f��?�Unknown
�HostRealDiv"^gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/RealDiv_1(11�ZdT@91�ZdT@A1�ZdT@I1�ZdT@a6�'�U?iQ�1<��?�Unknown
�HostRealDiv"\gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/RealDiv(1���S�]R@9���S�]R@A���S�]R@I���S�]R@a��A��S?iN@�����?�Unknown
YHostPow"Adam/Pow(1�K7�A(R@9�K7�A(R@A�K7�A(R@I�K7�A(R@aDKPe�KS?it誠���?�Unknown
} HostSum",gradient_tape/model/likelihood_layer/mul/Sum(1�V�Q@9�V�Q@A�V�Q@I�V�Q@a`M� �R?i�^����?�Unknown
�!HostRealDiv"`gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1/RealDiv_2(1�E����Q@9�E����Q@A�E����Q@I�E����Q@a)�l���R?i�~��?�Unknown
�"HostSquare"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Square_1(1�E����Q@9�E����A@A�E����Q@I�E����A@a)�l���R?i��r����?�Unknown
w#HostSum"&gradient_tape/model/gp_layer/mul_3/Sum(1}?5^�QQ@9}?5^�QQ@A}?5^�QQ@I}?5^�QQ@a��	�gR?iR�w���?�Unknown
�$HostNeg"Xgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/Neg(1Zd;�?Q@9Zd;�?Q@AZd;�?Q@IZd;�?Q@a�˸�TR?i�lJ�?�Unknown
�%HostRealDiv"`gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1/RealDiv_1(1���Mb�P@9���Mb�P@A���Mb�P@I���Mb�P@aIX���R?i��ކN�?�Unknown
X&HostSlice"Slice(1��C�lWP@9��C�lWP@A��C�lWP@I��C�lWP@aj�P�]Q?i�8�~��?�Unknown
�'HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatMul_1(1!�rh�EP@9!�rh�E@@A!�rh�EP@I!�rh�E@@aN(�V�JQ?io�2���?�Unknown
(HostMul".gradient_tape/model/likelihood_layer/mul_1/Mul(1w��/mO@9w��/mO@Aw��/mO@Iw��/mO@a�@L��P?i�G�z�&�?�Unknown
�)HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1��"��NN@9��"��NN@A��"��NN@I��"��NN@a8��@�P?iݫ-�	/�?�Unknown
e*HostSum"model/gp_layer/Sum_6(1%��C+N@9%��C+N@A%��C+N@I%��C+N@a��F�P?i�'��7�?�Unknown
�+HostStridedSlice"&model/likelihood_layer/strided_slice_4(1�ʡE��M@9�ʡE��M@A�ʡE��M@I�ʡE��M@aW٭���O?i>���>�?�Unknown
�,HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatMul(1w��/M@9w��/=@Aw��/M@Iw��/=@a#�w��N?idٶF�?�Unknown
�-HostStridedSlice"&model/likelihood_layer/strided_slice_3(1w��/M@9w��/M@Aw��/M@Iw��/M@a#�w��N?i��nN�?�Unknown
�.HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatMul_2(1�(\���L@9�(\���<@A�(\���L@I�(\���<@a�
����N?i�/��V�?�Unknown
e/HostSum"model/gp_layer/Sum_3(1H�z��K@9H�z��K@AH�z��K@IH�z��K@a�}�z�cM?i��D�l]�?�Unknown
�0HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_2(1�p=
׃K@9�p=
׃;@A�p=
׃K@I�p=
׃;@a�4���=M?i}�1@�d�?�Unknown
�1HostNeg"Tgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub/Neg(1��x�&�J@9��x�&�J@A��x�&�J@I��x�&�J@af���L?i��y<�k�?�Unknown
�2HostSelectV2"*model/gp_layer/LogNormal/log_prob/SelectV2(1}?5^��J@9}?5^��J@A}?5^��J@I}?5^��J@a���7	4L?io�>�r�?�Unknown
{3HostMatrixDiagV3"!gradient_tape/model/gp_layer/diag(1��v���I@9��v���I@A��v���I@I��v���I@a2	��K?i�2�;�y�?�Unknown
�4HostRealDiv"^gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/RealDiv_2(1���(\�I@9���(\�I@A���(\�I@I���(\�I@a�*�)K?iG}�����?�Unknown
y5HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1bX9��H@9bX9��H@AbX9��H@IbX9��H@au3e��EJ?i�7.,��?�Unknown
�6HostSoftplus"2model/gp_layer/truediv_2/softplus/forward/Softplus(1bX9��H@9bX9��H@AbX9��H@IbX9��H@au3e��EJ?i�p����?�Unknown
�7HostNeg"Vgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub_3/Neg(1!�rh�MH@9!�rh�MH@A!�rh�MH@I!�rh�MH@aʔ���I?i���2��?�Unknown
s8HostMul""gradient_tape/model/gp_layer/mul_2(1=
ףpMH@9=
ףpMH@A=
ףpMH@I=
ףpMH@a[ց8��I?i|�A����?�Unknown
�9HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_8_grad/TensorListPopBack(1}?5^�)H@9}?5^�)8@A}?5^�)H@I}?5^�)8@a"L�D�I?i�����?�Unknown
�:HostRealDiv"Hmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1(1;�O���G@9;�O���G@A;�O���G@I;�O���G@av��i<I?izgmb��?�Unknown
m;HostMul"model/likelihood_layer/mul_1(1���x�vG@9���x�vG@A���x�vG@I���x�vG@a'����H?iA�����?�Unknown
c<HostExp"model/gp_layer/Exp(133333SG@933333SG@A33333SG@I33333SG@a����H?i��ހг�?�Unknown
�=HostSub"Dmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub_3(1sh��|/G@9sh��|/G@Ash��|/G@Ish��|/G@a�x��H?i�:}���?�Unknown
g>HostAddN"Adam/gradients/AddN_5(1��n��F@9��n��F@A��n��F@I��n��F@aa �S�H?io7�|���?�Unknown
�?HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_23(1�����|F@9�����|6@A�����|F@I�����|6@a(�Q`
�G?i�KI����?�Unknown
�@HostMul"Vgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul_1/Mul(1�&1�|F@9�&1�|F@A�&1�|F@I�&1�|F@a��R���G?i��,y���?�Unknown
}AHostSelectV2"'gradient_tape/model/gp_layer/SelectV2_3(1h��|?5F@9h��|?5F@Ah��|?5F@Ih��|?5F@aF����G?i|$����?�Unknown
�BHostAddN"/Adam/gradients/PartitionedCall/gradients/AddN_4(1� �rhF@9� �rhF@A� �rhF@I� �rhF@a�zo��sG?i[ N���?�Unknown
sCHost	ZerosLike"Adam/gradients/zeros_like_26(1� �rhF@9� �rhF@A� �rhF@I� �rhF@a�zo��sG?i:܅���?�Unknown
�DHostMul"Bmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul(1� �rhF@9� �rhF@A� �rhF@I� �rhF@a�zo��sG?i���l��?�Unknown
�EHostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1�Q���R@9�Q���R@A����M�E@I����M�E@a���j��F?is��#��?�Unknown
�FHostMul"Pgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul(1�I+E@9�I+E@A�I+E@I�I+E@a16��iF?i�t|Z���?�Unknown
�GHostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/strided_slice_1(1�I+E@9�I+5@A�I+E@I�I+5@a16��iF?i�v`�X��?�Unknown
�HHostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_grad/MatMul(1�~j�t�D@9�~j�t�4@A�~j�t�D@I�~j�t�4@a��e�DF?i�G����?�Unknown
�IHostMul"Zgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1/mul(1�~j�t�D@9�~j�t�D@A�~j�t�D@I�~j�t�D@a��e�DF?i��.�z��?�Unknown
hJHostAddV2"model/gp_layer/add_10(1q=
ף@D@9q=
ף@D@Aq=
ף@D@Iq=
ף@D@a�{@4��E?i���X��?�Unknown
}KHostSelectV2"'gradient_tape/model/gp_layer/SelectV2_2(1�rh��D@9�rh��D@A�rh��D@I�rh��D@a��@`E?i[�KZ4
�?�Unknown
�LHostMul"Xgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/mul(1�����D@9�����D@A�����D@I�����D@aV3�m�_E?ihI'S��?�Unknown
�MHostNeg"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/sub_3_grad/Neg(1L7�A`�C@9L7�A`�3@AL7�A`�C@IL7�A`�3@a�]��E?i��R��?�Unknown
�NHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_6_grad/Shape/TensorListPopBack(1�l����C@9�l����3@A�l����C@I�l����3@a�����D?i�����?�Unknown
�OHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate(1�O��n�E@9�O��n�E@A�l����C@I�l����C@a�����D?iz>�UH�?�Unknown
�PHostRealDiv"Fmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv(1'1�jC@9'1�jC@A'1�jC@I'1�jC@a��y���D?i�\�p$�?�Unknown
~QHostMatMul"*gradient_tape/model/gp_layer/MatMul/MatMul(1D�l��iC@9D�l��iC@AD�l��iC@ID�l��iC@a[{ڡD?i���K�)�?�Unknown
uRHostFlushSummaryWriter"FlushSummaryWriter(1��(\�"C@9��(\�"C@A��(\�"C@I��(\�"C@a��7�UD?i�IȮ.�?�Unknown�
�SHostMul"4gradient_tape/model/likelihood_layer/exp/forward/mul(1��(\�"C@9��(\�"C@A��(\�"C@I��(\�"C@a��7�UD?i�טD�3�?�Unknown
�THost	ZerosLike"Wgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/zeros_like(1��(\�"C@9��(\�"C@A��(\�"C@I��(\�"C@a��7�UD?i�e ��8�?�Unknown
�UHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_7_grad/TensorListPopBack(1���K�B@9���K�2@A���K�B@I���K�2@aΑTp��C?i�z���=�?�Unknown
kVHostMatMul"model/gp_layer/MatMul_1(19��v�oB@99��v�oB@A9��v�oB@I9��v�oB@a���C?is�)��B�?�Unknown
}WHostSelectV2"'gradient_tape/model/gp_layer/SelectV2_1(1y�&1LB@9y�&1LB@Ay�&1LB@Iy�&1LB@a�4q��qC?i��Z5�G�?�Unknown
�XHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/triangular_solve/MatrixTriangularSolve/TensorListPopBack(1�G�zB@9�G�z2@A�G�zB@I�G�z2@a�a/�%C?i��ܯ^L�?�Unknown
�YHostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_13(1T㥛��A@9T㥛��1@AT㥛��A@IT㥛��1@a�׍��B?i�a�Q�?�Unknown
gZHostAddN"Adam/gradients/AddN_8(1q=
ף�A@9q=
ף�A@Aq=
ף�A@Iq=
ף�A@a,�A��B?i�.2��U�?�Unknown
g[HostAddN"Adam/gradients/AddN_7(1�rh��A@9�rh��A@A�rh��A@I�rh��A@a��M��B?i8��Z�?�Unknown
�\HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_grad/Mul(1�rh��A@9�rh��1@A�rh��A@I�rh��1@a��M��B?i�%ّK_�?�Unknown
�]HostSquare"smodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Square(1�rh��A@9�rh��1@A�rh��A@I�rh��1@a��M��B?i ��	d�?�Unknown
�^HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_15(1�MbX�A@9�MbX�1@A�MbX�A@I�MbX�1@a'�J-�B?i��7�h�?�Unknown
�_HostBroadcastTo"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_1_grad/BroadcastTo(1+��A@9+��1@A+��A@I+��1@aJFM�ȳB?i�Y�[m�?�Unknown
�`HostStridedSliceGrad"cgradient_tape/model/gp_layer/fill_triangular/forward/fill_triangular/strided_slice/StridedSliceGrad(1�l���QA@9�l���QA@A�l���QA@I�l���QA@a�1
��gB?i�Ɂ��q�?�Unknown
�aHostMatMul"smodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/MatMul(1�l���QA@9�l���Q1@A�l���QA@I�l���Q1@a�1
��gB?i̩�v�?�Unknown
�bHostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem_grad/TensorListReserve(1ˡE��-A@9ˡE��-1@AˡE��-A@IˡE��-1@a��h��AB?iE��j {�?�Unknown
�cHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_8_grad/Shape/TensorListPopBack(1ˡE��-A@9ˡE��-1@AˡE��-A@IˡE��-1@a��h��AB?io  ��?�Unknown
sdHost	ZerosLike"Adam/gradients/zeros_like_28(1ˡE��-A@9ˡE��-A@AˡE��-A@IˡE��-A@a��h��AB?i�+aA��?�Unknown
�eHostMatMul".gradient_tape/model/gp_layer/MatMul_1/MatMul_1(1��� ��@@9��� ��@@A��� ��@@I��� ��@@a�J����A?i컪\���?�Unknown
zfHostMul")gradient_tape/model/gp_layer/mul_26/Mul_1(1��� ��@@9��� ��@@A��� ��@@I��� ��@@a�J����A?i?]*X)��?�Unknown
�gHostMatMul"Mmodel/gp_layer/Tensordot_1/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1��"���@@9��"���@@A��"���@@I��"���@@aK��
��A?i/�֓��?�Unknown
fhHostSum"model/gp_layer/Sum_12(1B`��"{@@9B`��"{@@AB`��"{@@IB`��"{@@a�wCD�A?i'~����?�Unknown
�iHostMatMul",gradient_tape/model/gp_layer/MatMul_1/MatMul(1�ʡE�3@@9�ʡE�3@@A�ʡE�3@@I�ʡE�3@@a2c ]�7A?i&gU�B��?�Unknown
ejHostMul"model/gp_layer/mul_4(1�ʡE�3@@9�ʡE�3@@A�ʡE�3@@I�ʡE�3@@a2c ]�7A?i?�,͐��?�Unknown
�kHost
Reciprocal"Wgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/Reciprocal(1�$��3@@9�$��3@@A�$��3@@I�$��3@@a¤��7A?i�'O�ޢ�?�Unknown
�lHostTensorListLength"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength(1���Q�?@9���Q�/@A���Q�?@I���Q�/@aP�����@?iL�w���?�Unknown
dmHostDataset"Iterator::Model(1�MbXIO@9�MbXIO@A���Q�?@I���Q�?@aP�����@?i����T��?�Unknown
hnHostPack"model/gp_layer/stack_2(1���Q�?@9���Q�?@A���Q�?@I���Q�?@aP�����@?i�6ɷ���?�Unknown
�oHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1����xI?@9����xI/@A����xI?@I����xI/@a�{{��@?is�����?�Unknown
spHostDataset"Iterator::Model::ParallelMapV2(1�|?5^�>@9�|?5^�>@A�|?5^�>@I�|?5^�>@a��9�S@?i�cx�̷�?�Unknown
rqHostTensorSliceDataset"TensorSliceDataset(1�|?5^�>@9�|?5^�>@A�|?5^�>@I�|?5^�>@a��9�S@?iG�����?�Unknown
�rHostMul"Dmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul_1(1'1��>@9'1��>@A'1��>@I'1��>@a��:.�S@?iAį���?�Unknown
�sHostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_3/TensorListStack(1��� �r>@9��� �r>@A��� �r>@I��� �r>@aU`�:�-@?iZ�)��?�Unknown
�tHostSum"pmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Sum(1��� �r>@9��� �r.@A��� �r>@I��� �r.@aU`�:�-@?i��a���?�Unknown
�uHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_grad/MatMul/TensorListPopBack(1j�t��=@9j�t��-@Aj�t��=@Ij�t��-@a��LD�??i�"�+��?�Unknown
�vHostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem_grad/TensorListSetItem(1�p=
��=@9�p=
��-@A�p=
��=@I�p=
��-@aŗ����??i�������?�Unknown
swHostMul""gradient_tape/model/gp_layer/Mul_5(1�p=
��=@9�p=
��=@A�p=
��=@I�p=
��=@aŗ����??iΔ+���?�Unknown
{xHostSelectV2"%gradient_tape/model/gp_layer/SelectV2(1�p=
��=@9�p=
��=@A�p=
��=@I�p=
��=@aŗ����??i��i����?�Unknown
�yHostRealDiv"0gradient_tape/model/gp_layer/truediv_4/RealDiv_2(1\���(�=@9\���(�=@A\���(�=@I\���(�=@avl�w??i/ь����?�Unknown
�zHost	Transpose"<gradient_tape/model/gp_layer/Tensordot_1/transpose/transpose(1��~j�T=@9��~j�T=@A��~j�T=@I��~j�T=@a�(2�+??iM�"���?�Unknown
�{HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_5(1Zd;�O=@9Zd;�O-@AZd;�O=@IZd;�O-@a���J�>?i	s�"���?�Unknown
�|HostStridedSlice"|model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/strided_slice_3(1X9��v~<@9X9��v~,@AX9��v~<@IX9��v~,@a��_|1H>?i �)i��?�Unknown
u}HostTile"#gradient_tape/model/gp_layer/Tile_8(1��|?5~<@9��|?5~<@A��|?5~<@I��|?5~<@a�7b��G>?iGˆ&2��?�Unknown
n~HostSigmoid"Adam/gradients/Sigmoid_12(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a\#��=?i+�����?�Unknown
�HostSelectV2"<gradient_tape/model/gp_layer/LogNormal_2/log_prob/SelectV2_1(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a\#��=?i��'���?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_1/RealDiv_2(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a\#��=?i�v �p��?�Unknown
��HostAddV2"Amodel/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Internal_0_add_1(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a\#��=?i�Z~(0��?�Unknown
f�HostMul"Adam/gradients/mul_7(1J+�6<@9J+�6<@AJ+�6<@IJ+�6<@a��!I��=?iG����?�Unknown
x�HostMul"&gradient_tape/model/gp_layer/mul_7/Mul(1���(\�;@9���(\�;@A���(\�;@I���(\�;@a���=?i�zȣ��?�Unknown
��HostAddN"/Adam/gradients/PartitionedCall/gradients/AddN_1(1�v���;@9�v���;@A�v���;@I�v���;@a��a֯=?i`���[�?�Unknown
��HostStridedSliceGrad">gradient_tape/model/gp_layer/strided_slice_13/StridedSliceGrad(1H�z��;@9H�z��;@AH�z��;@IH�z��;@a�}�z�c=?i�	d	�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_3(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a)iX�=?i�t6��?�Unknown
��HostUnpack",gradient_tape/model/gp_layer/stack_2/unstack(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a)iX�=?i��N�?�Unknown
g�HostMul"model/gp_layer/mul_10(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a)iX�=?i�J���?�Unknown
��HostStridedSlice"&model/likelihood_layer/strided_slice_8(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a)iX�=?i����?�Unknown
��HostSelectV2"*model/gp_layer/softplus_6/forward/SelectV2(1F����;@9F����;@AF����;@IF����;@a�T��<?i�8��-�?�Unknown
f�HostExp"model/gp_layer/Exp_3(1j�t�;@9j�t�;@Aj�t�;@Ij�t�;@a����<?i�����?�Unknown
��HostAddV2"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/add_2(1j�t�;@9j�t�+@Aj�t�;@Ij�t�+@a����<?i��ę`"�?�Unknown
��HostDynamicStitch"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_1_grad/DynamicStitch(1� �rh�:@9� �rh�*@A� �rh�:@I� �rh�*@aC@��6�<?i�X���%�?�Unknown
��HostAddN"-Adam/gradients/PartitionedCall/gradients/AddN(1����x�:@9����x�:@A����x�:@I����x�:@a2���3<?is�w)�?�Unknown
l�HostConcatV2"model/gp_layer/concat(1����MB:@9����MB:@A����MB:@I����MB:@a��NP!�;?iF�9�,�?�Unknown
��HostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_2/TensorListStack(15^�IB:@95^�IB:@A5^�IB:@I5^�IB:@a�Q���;?ij߮q0�?�Unknown
f�HostExp"model/gp_layer/Exp_2(15^�IB:@95^�IB:@A5^�IB:@I5^�IB:@a�Q���;?i�)$�3�?�Unknown
��HostBroadcastTo"(gradient_tape/model/gp_layer/BroadcastTo(133333�9@933333�9@A33333�9@I33333�9@a����P;?i�X7�?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_1(133333�9@933333�)@A33333�9@I33333�)@a����P;?iL�:�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_22(1m����9@9m����)@Am����9@Im����)@a�w�5�O;?i���,>�?�Unknown
g�HostMul"model/gp_layer/mul_17(1m����9@9m����9@Am����9@Im����9@a�w�5�O;?i����A�?�Unknown
��HostRealDiv".gradient_tape/model/gp_layer/truediv/RealDiv_1(1����k9@9����k9@A����k9@I����k9@aM���#;?i� '��D�?�Unknown
n�HostSigmoid"Adam/gradients/Sigmoid_7(1�Q��k9@9�Q��k9@A�Q��k9@I�Q��k9@aqc�N�;?i���WH�?�Unknown
��HostMatMul"Kmodel/gp_layer/Tensordot/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1j�t�$9@9j�t�$9@Aj�t�$9@Ij�t�$9@a�NGg��:?i�ڽ�K�?�Unknown
i�HostAddV2"model/gp_layer/add_22(1j�t�$9@9j�t�$9@Aj�t�$9@Ij�t�$9@a�NGg��:?i�ÊO�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_14(1�&1��8@9�&1��(@A�&1��8@I�&1��(@a�:�l:?iM�Z�RR�?�Unknown
��HostStridedSliceGrad"rgradient_tape/model/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/strided_slice/StridedSliceGrad(1�&1��8@9�&1��8@A�&1��8@I�&1��8@a�:�l:?i��*
�U�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/add_grad/Sum(1#��~j�8@9#��~j�(@A#��~j�8@I#��~j�(@a����k:?i�F��X�?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_1_grad/Mul_1(1h��|?�8@9h��|?�(@Ah��|?�8@Ih��|?�(@a&��& :?i��1\�?�Unknown
v�HostSum"$gradient_tape/model/gp_layer/add/Sum(1�E����8@9�E����8@A�E����8@I�E����8@a<����:?iFv7�u_�?�Unknown
i�HostLess"Adam/gradients/Less_11(1!�rh�M8@9!�rh�M8@A!�rh�M8@I!�rh�M8@aʔ���9?iY�X�b�?�Unknown
��HostTensorListStack"7model/gp_layer/map/TensorArrayV2Stack_1/TensorListStack(1!�rh�M8@9!�rh�M8@A!�rh�M8@I!�rh�M8@aʔ���9?ilVz��e�?�Unknown
��HostBroadcastTo"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_grad/BroadcastTo(1Zd;�OM8@9Zd;�OM(@AZd;�OM8@IZd;�OM(@a��e��9?i���$i�?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem(1Zd;�OM8@9Zd;�OM(@AZd;�OM8@IZd;�OM(@a��e��9?i2�So_l�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_18(1���S�8@9���S�(@A���S�8@I���S�(@az@~ˇ9?i2�h�o�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_grad/Shape/TensorListPopBack(1X9��v�7@9X9��v�'@AX9��v�7@IX9��v�'@a����;9?i�^6�r�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_25/Mul(1ףp=
w7@9ףp=
w7@Aףp=
w7@Iףp=
w7@a�ڹ���8?iV���u�?�Unknown
l�HostSum"model/likelihood_layer/Sum(1ףp=
w7@9ףp=
w7@Aףp=
w7@Iףp=
w7@a�ڹ���8?iFM"��x�?�Unknown
��HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate(1     @@9     @@AZd;�/7@IZd;�/7@a CtnY�8?i�Po|�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/transpose_grad/InvertPermutation/TensorListPopBack(1V-��/7@9V-��/'@AV-��/7@IV-��/'@a#�v��8?i�*���?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_18/Sum(1V-��/7@9V-��/7@AV-��/7@IV-��/7@a#�v��8?i�9Bt1��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_22_grad/Shape/TensorListPopBack(1���(\/7@9���(\/'@A���(\/7@I���(\/'@aFIy"Σ8?i���E��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem/TensorListPopBack(1�x�&1�6@9�x�&1�&@A�x�&1�6@I�x�&1�&@a��3�+X8?i���P��?�Unknown
^�HostCast"Adam/Cast_1(1-����6@9-����6@A-����6@I-����6@a�46;�W8?i�J�[��?�Unknown
��HostTensorListGetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem(1-����6@9-����&@A-����6@I-����&@a�46;�W8?i�|�f��?�Unknown
f�HostExp"Adam/gradients/Exp_1(1��n��6@9��n��6@A��n��6@I��n��6@aa �S�8?i��lh��?�Unknown
��HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1��n��6@9��n��6@A��n��6@I��n��6@aa �S�8?iuy��i��?�Unknown
��HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1��n��6@9��n��&@A��n��6@I��n��&@aa �S�8?i��plk��?�Unknown
��HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_grad/MatMul_1(1+�Y6@9+�Y&@A+�Y6@I+�Y&@a��l�7?iڍ>oc��?�Unknown
��HostSelectV2"*model/gp_layer/softplus_7/forward/SelectV2(1F����X6@9F����X6@AF����X6@IF����X6@a���п7?i,dWi[��?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1� �rh6@9� �rh6@A� �rh6@I� �rh6@a�zo��s7?iRs�I��?�Unknown
��HostSelectV2"2model/gp_layer/truediv_5/softplus/forward/SelectV2(1� �rh6@9� �rh6@A� �rh6@I� �rh6@a�zo��s7?i
@�c8��?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_4(1}?5^��5@9}?5^��%@A}?5^��5@I}?5^��%@aP�.R�'7?i��Z��?�Unknown
n�HostSelectV2"Adam/gradients/SelectV2(1��(\��5@9��(\��5@A��(\��5@I��(\��5@a�Q��6?i�����?�Unknown
��HostMatMul"bgradient_tape/model/gp_layer/Tensordot_1/MatMul/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1��(\��5@9��(\��5@A��(\��5@I��(\��5@a�Q��6?i;�=aԫ�?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_1_grad/Mul(1����M�5@9����M�%@A����M�5@I����M�%@a���j��6?i��ۯ��?�Unknown
��Host
Reciprocal"cgradient_tape/model/gp_layer/LogNormal/log_prob/LogNormal_exp_2/inverse_log_det_jacobian/Reciprocal(1����M�5@9����M�5@A����M�5@I����M�5@a���j��6?i1{V���?�Unknown
f�HostSum"model/gp_layer/Sum_8(1����M�5@9����M�5@A����M�5@I����M�5@a���j��6?i�؅�f��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_13_grad/TensorListPopBack(1{�G�:5@9{�G�:%@A{�G�:5@I{�G�:%@ak����6?i�M��8��?�Unknown
��Host
Reciprocal"Pgradient_tape/model/gp_layer/LogNormal/log_prob/LogNormal_exp/inverse/Reciprocal(1{�G�:5@9{�G�:5@A{�G�:5@I{�G�:5@ak����6?i��f�
��?�Unknown
g�HostAdd"model/gp_layer/Add_15(1{�G�:5@9{�G�:5@A{�G�:5@I{�G�:5@ak����6?i�7��ܼ�?�Unknown
f�HostMul"Adam/gradients/mul_1(1�~j�t�4@9�~j�t�4@A�~j�t�4@I�~j�t�4@a��e�D6?i��JI���?�Unknown
��HostStridedSliceGrad">gradient_tape/model/gp_layer/strided_slice_14/StridedSliceGrad(1�~j�t�4@9�~j�t�4@A�~j�t�4@I�~j�t�4@a��e�D6?i^Q��m��?�Unknown
{�HostAddV2"'model/gp_layer/LogNormal_3/log_prob/add(1�~j�t�4@9�~j�t�4@A�~j�t�4@I�~j�t�4@a��e�D6?i�1J6��?�Unknown
��HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1�/�$�E@9�/�$�E@A���ƫ4@I���ƫ4@a�%��5?i���D���?�Unknown
��HostRealDiv"]gradient_tape/model/gp_layer/LogNormal_2/log_prob/LogNormal_Normal_1/log_prob/truediv/RealDiv(1���ƫ4@9���ƫ4@A���ƫ4@I���ƫ4@a�%��5?iY��?���?�Unknown
o�HostSigmoid"Adam/gradients/Sigmoid_10(11�Zd4@91�Zd4@A1�Zd4@I1�Zd4@a6�'�5?i��z�i��?�Unknown
��HostMul"Sgradient_tape/model/gp_layer/LogNormal_1/log_prob/LogNormal_Normal_1/log_prob/mul_1(11�Zd4@91�Zd4@A1�Zd4@I1�Zd4@a6�'�5?i۟?;��?�Unknown
��HostSelectV2"<gradient_tape/model/gp_layer/LogNormal_3/log_prob/SelectV2_1(11�Zd4@91�Zd4@A1�Zd4@I1�Zd4@a6�'�5?i�����?�Unknown
��HostAddV2"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/add_1(11�Zd4@91�Zd$@A1�Zd4@I1�Zd$@a6�'�5?i]��6���?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_25(1�rh��4@9�rh��4@A�rh��4@I�rh��4@a��@`5?i;��76��?�Unknown
��HostSelectV2":gradient_tape/model/gp_layer/LogNormal/log_prob/SelectV2_1(1/�$��3@9/�$��3@A/�$��3@I/�$��3@aQ�[Y5?i��\����?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/mul_1(1h��|?�3@9h��|?�#@Ah��|?�3@Ih��|?�#@at`^��5?i�Cs6{��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_16(1�G�z�3@9�G�z�#@A�G�z�3@I�G�z�#@a��r6�4?i��A=��?�Unknown
j�HostWriteSummary"WriteSummary(1����ҍ3@9����ҍ3@A����ҍ3@I����ҍ3@aL���4?i
[;���?�Unknown�
|�HostNeg"*gradient_tape/model/gp_layer/truediv_2/Neg(1����ҍ3@9����ҍ3@A����ҍ3@I����ҍ3@aL���4?ip�t9F��?�Unknown
��HostAddN"=model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_add_8(1����ҍ3@9����ҍ3@A����ҍ3@I����ҍ3@aL���4?i��7���?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_9(1fffffF3@9fffffF#@AfffffF3@IfffffF#@a�7��|4?i᫪�n��?�Unknown
��Host	Transpose"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/adjoint/matrix_transpose/transpose(1fffffF3@9fffffF#@AfffffF3@IfffffF#@a�7��|4?i�F�9���?�Unknown
��HostConcatV2"Lmodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/concat(1fffffF3@9fffffF3@AfffffF3@IfffffF3@a�7��|4?i��㺍��?�Unknown
��HostStridedSlice"Smodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/strided_slice(1fffffF3@9fffffF3@AfffffF3@IfffffF3@a�7��|4?i�| <��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_13(1fffffF3@9fffffF#@AfffffF3@IfffffF#@a�7��|4?i�����?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_17(1��"���2@9��"���"@A��"���2@I��"���"@a#�� 04?i��<�2��?�Unknown
��HostDynamicStitch"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_grad/DynamicStitch(1��"���2@9��"���"@A��"���2@I��"���"@a#�� 04?iE}\Ÿ��?�Unknown
p�HostSelectV2"Adam/gradients/SelectV2_2(1��Q��2@9��Q��2@A��Q��2@I��Q��2@aA��W�/4?i:p��>��?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_19/Sum(1���K�2@9���K�2@A���K�2@I���K�2@aΑTp��3?i�z5?���?�Unknown
��HostTensorListFromTensor":model/gp_layer/map/TensorArrayUnstack/TensorListFromTensor(1Zd;�o2@9Zd;�o2@AZd;�o2@IZd;�o2@a\}��3?i���@.�?�Unknown
n�HostRealDiv"model/gp_layer/truediv_3(1Zd;�o2@9Zd;�o2@AZd;�o2@IZd;�o2@a\}��3?i,�B��?�Unknown
��HostMatMul"`gradient_tape/model/gp_layer/Tensordot/MatMul/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1V-��o2@9V-��o2@AV-��o2@IV-��o2@a �ŗ3?i�!�:�?�Unknown
x�HostMul"&gradient_tape/model/gp_layer/mul_1/Mul(1V-��o2@9V-��o2@AV-��o2@IV-��o2@a �ŗ3?i,��3��?�Unknown
g�HostMul"model/gp_layer/mul_16(1�x�&1(2@9�x�&1(2@A�x�&1(2@I�x�&1(2@a����K3?iJ�O��
�?�Unknown
�HostEqual"+model/likelihood_layer/assert_equal_3/Equal(1�x�&1(2@9�x�&1(2@A�x�&1(2@I�x�&1(2@a����K3?ihx+Z�?�Unknown
��HostStridedSlice"&model/likelihood_layer/strided_slice_7(1�x�&1(2@9�x�&1(2@A�x�&1(2@I�x�&1(2@a����K3?i��Φ��?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_16(1T㥛��1@9T㥛��!@AT㥛��1@IT㥛��!@a�׍��2?iA���#�?�Unknown
��HostTensorListStack"Cgradient_tape/model/gp_layer/map/TensorArrayUnstack/TensorListStack(1T㥛��1@9T㥛��1@AT㥛��1@IT㥛��1@a�׍��2?i�T���?�Unknown
��HostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_1/TensorListStack(1T㥛��1@9T㥛��1@AT㥛��1@IT㥛��1@a�׍��2?i�����?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_17/Mul(1�MbX�1@9�MbX�1@A�MbX�1@I�MbX�1@a'�J-�2?iQ�$:�?�Unknown
g�HostAdd"model/gp_layer/Add_11(1+��1@9+��1@A+��1@I+��1@aJFM�ȳ2?i�:흐�?�Unknown
��HostTensorListGetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem(1+��1@9+��!@A+��1@I+��!@aJFM�ȳ2?ia$���?�Unknown
n�HostRealDiv"model/gp_layer/truediv_4(1+��1@9+��1@A+��1@I+��1@aJFM�ȳ2?i
�= �?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_14_grad/Shape/TensorListPopBack(1�l���Q1@9�l���Q!@A�l���Q1@I�l���Q!@a�1
��g2?iP#��"�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_grad/TensorListPopBack(1�l���Q1@9�l���Q!@A�l���Q1@I�l���Q!@a�1
��g2?i�7��$�?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_4/TensorListFromTensor(1�l���Q1@9�l���Q1@A�l���Q1@I�l���Q1@a�1
��g2?i�K�$'�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_17_grad/Shape/TensorListPopBack(1
ףp=
1@9
ףp=
!@A
ףp=
1@I
ףp=
!@aeǸ�2?i�*bh)�?�Unknown
p�HostSelectV2"Adam/gradients/SelectV2_4(1
ףp=
1@9
ףp=
1@A
ףp=
1@I
ףp=
1@aeǸ�2?i�Cy��+�?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_5/RealDiv_1(1
ףp=
1@9
ףp=
1@A
ףp=
1@I
ףp=
1@aeǸ�2?i�\��-�?�Unknown
��HostSelectV2"2model/gp_layer/truediv_2/softplus/forward/SelectV2(1
ףp=
1@9
ףp=
1@A
ףp=
1@I
ףp=
1@aeǸ�2?ilu��20�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_2_grad/Shape/TensorListPopBack(1D�l��	1@9D�l��	!@AD�l��	1@ID�l��	!@a����2?i��	�u2�?�Unknown
z�HostStridedSlice"model/gp_layer/strided_slice_13(1�A`���0@9�A`���0@A�A`���0@I�A`���0@a����1?i!�#��4�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_22_grad/TensorListPopBack(1��(\��0@9��(\�� @A��(\��0@I��(\�� @a��+��1?i�o���6�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_34(1�Zd{0@9�Zd{0@A�Zd{0@I�Zd{0@a��@�(�1?i��w9�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_20(1B`��"{0@9B`��"{ @AB`��"{0@IB`��"{ @a�wCD�1?i�@�J;�?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_2/TensorListGetItem_grad/TensorListElementShape(1B`��"{0@9B`��"{ @AB`��"{0@IB`��"{ @a�wCD�1?i��wp{=�?�Unknown
o�HostSigmoid"Adam/gradients/Sigmoid_14(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@a�wCD�1?i_Q��?�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_29(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@a�wCD�1?i��Hi�A�?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_20(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@a�wCD�1?i=b��D�?�Unknown
��HostSelectV2"*model/gp_layer/softplus_1/forward/SelectV2(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@a�wCD�1?i��b=F�?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1�ʡE�30@9�ʡE�30@A�ʡE�30@I�ʡE�30@a2c ]�71?i���adH�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_3_grad/Sum(1�ʡE�30@9�ʡE�3 @A�ʡE�30@I�ʡE�3 @a2c ]�71?i�*�`�J�?�Unknown
s�HostConcatenateDataset"ConcatenateDataset(1�ʡE�30@9�ʡE�30@A�ʡE�30@I�ʡE�30@a2c ]�71?i��\`�L�?�Unknown
��HostMatMul",gradient_tape/model/gp_layer/MatMul/MatMul_1(1�ʡE�30@9�ʡE�30@A�ʡE�30@I�ʡE�30@a2c ]�71?i�j�_�N�?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_3(1�ʡE�30@9�ʡE�3 @A�ʡE�30@I�ʡE�3 @a2c ]�71?i�
4_ Q�?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_1(1j�t��/@9j�t��/@Aj�t��/@Ij�t��/@a�N�u�0?i�¢�S�?�Unknown
��HostSelectV2"<gradient_tape/model/gp_layer/LogNormal_1/log_prob/SelectV2_1(1j�t��/@9j�t��/@Aj�t��/@Ij�t��/@a�N�u�0?i<zd;U�?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_1/TensorListFromTensor(1j�t��/@9j�t��/@Aj�t��/@Ij�t��/@a�N�u�0?i�1��XW�?�Unknown
��HostMatrixBandPart"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatrixBandPart(1��Mb�/@9��Mb�@A��Mb�/@I��Mb�@a�ѿ���0?i�):`vY�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_26/Mul(1��Mb�/@9��Mb�/@A��Mb�/@I��Mb�/@a�ѿ���0?i�!�ٓ[�?�Unknown
��HostSelectV2",model/gp_layer/LogNormal_1/log_prob/SelectV2(1��Mb�/@9��Mb�/@A��Mb�/@I��Mb�/@a�ѿ���0?i��S�]�?�Unknown
��HostMatMul"6gradient_tape/model/gp_layer/Tensordot/MatMul/MatMul_1(1}?5^�I/@9}?5^�I/@A}?5^�I/@I}?5^�I/@aL:z�+�0?i�Y�_�?�Unknown
��HostMatMul"8gradient_tape/model/gp_layer/Tensordot_1/MatMul/MatMul_1(1}?5^�I/@9}?5^�I/@A}?5^�I/@I}?5^�I/@aL:z�+�0?ib��^�a�?�Unknown
}�HostEqual")model/gp_layer/LogNormal_3/log_prob/Equal(1}?5^�I/@9}?5^�I/@A}?5^�I/@I}?5^�I/@aL:z�+�0?i��d�c�?�Unknown
f�HostAddN"Adam/gradients/AddN(1��K7I/@9��K7I/@A��K7I/@I��K7I/@ao�|��0?iA��`f�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_18(1��K7I/@9��K7I/@A��K7I/@I��K7I/@ao�|��0?i٦}]h�?�Unknown
��HostGreaterEqual")gradient_tape/model/gp_layer/GreaterEqual(1��K7I/@9��K7I/@A��K7I/@I��K7I/@ao�|��0?iq�:Z)j�?�Unknown
v�HostTile"#gradient_tape/model/gp_layer/Tile_9(1��K7I/@9��K7I/@A��K7I/@I��K7I/@ao�|��0?i	��V=l�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_12(1�|?5^�.@9�|?5^�@A�|?5^�.@I�|?5^�@a��9�S0?i>��Gn�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_21(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@a��9�S0?isxVRp�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_5(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@a��9�S0?i�;8�\r�?�Unknown
��HostSelectV2",model/gp_layer/LogNormal_3/log_prob/SelectV2(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@a��9�S0?i�b�Ugt�?�Unknown
j�HostMatMul"model/gp_layer/MatMul(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@a��9�S0?i���qv�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_1_grad/BroadcastGradientArgs/TensorListPopBack(1�Q��+.@9�Q��+@A�Q��+.@I�Q��+@a���0?i��{�rx�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_24(1�Q��+.@9�Q��+.@A�Q��+.@I�Q��+.@a���0?i�?�sz�?�Unknown
��HostAddV2"Amodel/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Internal_0_add_6(1�Q��+.@9�Q��+.@A�Q��+.@I�Q��+.@a���0?i�F�t|�?�Unknown
��HostSelectV2"2model/gp_layer/truediv_1/softplus/forward/SelectV2(1�Q��+.@9�Q��+.@A�Q��+.@I�Q��+.@a���0?i^���u~�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_7(1^�I+.@9^�I+@A^�I+.@I^�I+@a��s�0?i���v��?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_grad/Mul_1(1^�I+.@9^�I+@A^�I+.@I^�I+@a��s�0?i����w��?�Unknown
n�HostSigmoid"Adam/gradients/Sigmoid_1(1^�I+.@9^�I+.@A^�I+.@I^�I+.@a��s�0?i���x��?�Unknown
��HostSelectV2",model/gp_layer/LogNormal_2/log_prob/SelectV2(1^�I+.@9^�I+.@A^�I+.@I^�I+.@a��s�0?i���y��?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_8(1�&1��-@9�&1��@A�&1��-@I�&1��@a/ ge\x/?iZ��Nq��?�Unknown
n�HostMaximum"model/gp_layer/Maximum_3(1�&1��-@9�&1��-@A�&1��-@I�&1��-@a/ ge\x/?i�.��h��?�Unknown
f�HostSum"model/gp_layer/Sum_2(1�&1��-@9�&1��-@A�&1��-@I�&1��-@a/ ge\x/?i:�RZ`��?�Unknown
f�HostExp"model/gp_layer/Exp_1(1\���(�-@9\���(�-@A\���(�-@I\���(�-@avl�w/?i�d�W��?�Unknown
��HostSum"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Sum_1(1�����-@9�����@A�����-@I�����@aJ�����.?i�-�E��?�Unknown
f�HostExp"Adam/gradients/Exp_2(1Zd;�O-@9Zd;�O-@AZd;�O-@IZd;�O-@a���J�.?ie8B�3��?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�����-@9�����-@A�����-@I�����-@a����u�.?i'��!��?�Unknown
��Host	ReverseV2"Omodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/ReverseV2(1X9��v~,@9X9��v~,@AX9��v~,@IX9��v~,@a��_|1H.?i�Z��?�Unknown
i�HostPack"model/gp_layer/stack_1(1X9��v~,@9X9��v~,@AX9��v~,@IX9��v~,@a��_|1H.?i	�����?�Unknown
m�HostIteratorGetNext"IteratorGetNext(1ˡE��},@9ˡE��},@AˡE��},@IˡE��},@a�d0�G.?iU�4Xϙ�?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve(1�v���+@9�v���@A�v���+@I�v���@a��a֯-?i>ךU���?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem_grad/TensorListElementShape(1�v���+@9�v���@A�v���+@I�v���@a��a֯-?i'� S���?�Unknown
��HostLess"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/cond/_311/gradients/model/gp_layer/map/while_grad/Less(1�v���+@9��)g�@A�v���+@I��)g�@a��a֯-?igP`��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_16(1�v���+@9�v���+@A�v���+@I�v���+@a��a֯-?i�0�M;��?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_2/TensorListFromTensor(1�v���+@9�v���+@A�v���+@I�v���+@a��a֯-?i�N3K��?�Unknown
��HostAssert"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/Assert/Assert(1�v���+@9�v���@A�v���+@I�v���@a��a֯-?i�l�H��?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_2(1�v���+@9�v���@A�v���+@I�v���@a��a֯-?i���E̦�?�Unknown
��HostSelectV2"2model/gp_layer/truediv_4/softplus/forward/SelectV2(1�v���+@9�v���+@A�v���+@I�v���+@a��a֯-?i��eC���?�Unknown
��HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a)iX�-?i$���x��?�Unknown
{�HostMul")gradient_tape/model/gp_layer/mul_12/Mul_1(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a)iX�-?i�8DJ��?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_24/Sum(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a)iX�-?i2I����?�Unknown
l�HostMaximum"model/gp_layer/Maximum(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a)iX�-?i�~
E��?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(19��v�_+@99��v�_+@A9��v�_+@I9��v�_+@aoo]G{-?i�������?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_4/RealDiv_1(1� �rh�*@9� �rh�*@A� �rh�*@I� �rh�*@aC@��6�,?i�A+����?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_13(17�A`��*@97�A`��*@A7�A`��*@I7�A`��*@a�F�x�,?i(��N��?�Unknown
��HostReadVariableOp"6model/gp_layer/Reshape/identity/forward/ReadVariableOp(17�A`��*@97�A`��*@A7�A`��*@I7�A`��*@a�F�x�,?i�\����?�Unknown
��HostStridedSlice"|model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/strided_slice_1(17�A`��*@97�A`��@A7�A`��*@I7�A`��@a�F�x�,?i�Q�޸�?�Unknown
��HostMatrixBandPart"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatrixBandPart_grad/MatrixBandPart(1��(\�B*@9��(\�B@A��(\�B*@I��(\�B@a^L�f�+?i�N�6���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_6_grad/TensorListPopBack(1��(\�B*@9��(\�B@A��(\�B*@I��(\�B@a^L�f�+?i��0�[��?�Unknown
��HostGreaterEqual"+gradient_tape/model/gp_layer/GreaterEqual_3(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a�Q���+?i�X�:��?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_22(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a�Q���+?i����ؿ�?�Unknown
��HostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_4/TensorListStack(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a�Q���+?iȢ`6���?�Unknown
x�HostGatherV2"!model/gp_layer/Tensordot/GatherV2(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a�Q���+?i�G�U��?�Unknown
��HostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/strided_slice_2(15^�IB*@95^�IB@A5^�IB*@I5^�IB@a�Q���+?i���1��?�Unknown
{�HostLog1p"'model/gp_layer/softplus_7/forward/Log1p(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a�Q���+?i�������?�Unknown
��HostAdd"Dmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub_2(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a�Q���+?i7K-���?�Unknown
��HostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes/strided_slice_1(1��K7�A*@9��K7�A@A��K7�A*@I��K7�A@a�#V^P�+?irQ�O��?�Unknown
��HostLess"qmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/cond/_126/model/gp_layer/map/while/Less(1�ʡE��)@9�1��y"@A�ʡE��)@I�1��y"@ay��'�P+?iјë��?�Unknown
��HostAddN"8Adam/gradients/ArithmeticOptimizer/AddOpsRewrite_AddN_27(133333�)@933333�)@A33333�)@I33333�)@a����P+?i�U�����?�Unknown
h�HostLess"Adam/gradients/Less_2(133333�)@933333�)@A33333�)@I33333�)@a����P+?i/?�n��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_3_grad/TensorListPopBack(133333�)@933333�@A33333�)@I33333�@a����P+?i����#��?�Unknown
y�HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�&1�[�@9�&1�[�@A33333�)@I33333�)@a����P+?i�������?�Unknown
t�HostMul""gradient_tape/model/gp_layer/Mul_1(133333�)@933333�)@A33333�)@I33333�)@a����P+?i<Hx����?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_3(133333�)@933333�@A33333�)@I33333�@a����P+?i�6�B��?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_1_grad/Sum(1��� ��)@9��� ��@A��� ��)@I��� ��@a�Ϗ�O+?i�?����?�Unknown
x�HostSum"&gradient_tape/model/gp_layer/add/Sum_1(1��� ��)@9��� ��)@A��� ��)@I��� ��)@a�Ϗ�O+?i��G����?�Unknown
��HostConcatV2"=model/gp_layer/fill_triangular/forward/fill_triangular/concat(11�Z$)@91�Z$)@A1�Z$)@I1�Z$)@a��D<�*?i8�$X��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_16_grad/TensorListPopBack(1�p=
�#)@9�p=
�#@A�p=
�#)@I�p=
�#@a!�I���*?i�����?�Unknown
��HostSelectV2"2model/gp_layer/Squeeze_1/softplus/forward/SelectV2(1�p=
�#)@9�p=
�#)@A�p=
�#)@I�p=
�#)@a!�I���*?ir� ���?�Unknown
��Host	Transpose"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/adjoint/matrix_transpose/transpose_grad/transpose(1���S#)@9���S#@A���S#)@I���S#@ah�Nu%�*?i`Qx�Z��?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_1/RealDiv_1(1/�$��(@9/�$��(@A/�$��(@I/�$��(@a���>l *?iJ=<����?�Unknown
��HostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/strided_slice(1/�$��(@9/�$��@A/�$��(@I/�$��@a���>l *?i4) ����?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN(1�E����(@9�E����@A�E����(@I�E����@a<����*?ioU�@��?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_7(1�E����(@9�E����(@A�E����(@I�E����(@a<����*?i������?�Unknown
|�HostNeg"*gradient_tape/model/gp_layer/truediv_4/Neg(1�E����(@9�E����(@A�E����(@I�E����(@a<����*?i�-����?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_19(1�E����(@9�E����@A�E����(@I�E����@a<����*?i �<�&��?�Unknown
p�Host	Transpose"model/gp_layer/transpose(1�E����(@9�E����(@A�E����(@I�E����(@a<����*?i[L����?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_1_grad/MatMul_1/TensorListPopBack(1�G�z�(@9�G�z�@A�G�z�(@I�G�z�@a��ȦU*?i�r��j��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_7_grad/Shape/TensorListPopBack(1�G�z�(@9�G�z�@A�G�z�(@I�G�z�@a��ȦU*?iq� {��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_1(1-���(@9-���@A-���(@I-���@az8p��)?i������?�Unknown
^�HostCast"Adam/Cast_2(1�/�$(@9�/�$(@A�/�$(@I�/�$(@aW�=$�)?i�&څ=��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_35(1�/�$(@9�/�$(@A�/�$(@I�/�$(@aW�=$�)?i�j����?�Unknown
d�HostLog"model/gp_layer/Log(1�/�$(@9�/�$(@A�/�$(@I�/�$(@aW�=$�)?i����n��?�Unknown
��HostMul"Qgradient_tape/model/gp_layer/LogNormal_1/log_prob/LogNormal_Normal_1/log_prob/Mul(1��ʡ(@9��ʡ(@A��ʡ(@I��ʡ(@a��B؅�)?i�2\ ��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_11/Mul(1��ʡ(@9��ʡ(@A��ʡ(@I��ʡ(@a��B؅�)?iѶ�x���?�Unknown
\�HostPow"
Adam/Pow_1(1���Kw'@9���Kw'@A���Kw'@I���Kw'@arW�UA�(?iF�|.��?�Unknown
g�HostExp"Adam/gradients/Exp_10(1���Kw'@9���Kw'@A���Kw'@I���Kw'@arW�UA�(?i�m䀽��?�Unknown
i�HostAddN"Adam/gradients/AddN_11(1X9��v'@9X9��v'@AX9��v'@IX9��v'@a�]�	��(?i�	E|L��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_15(1X9��v'@9X9��v@AX9��v'@IX9��v@a�]�	��(?iG��w���?�Unknown
��HostPack"~model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_2/values_1(1X9��v'@9X9��v@AX9��v'@IX9��v@a�]�	��(?iAsj �?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_1/TensorListGetItem_grad/TensorListElementShape(1�� �r�&@9�� �r�@A�� �r�&@I�� �r�@a�.1�qX(?i ����?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem_grad/TensorListSetItem(1-����&@9-����@A-����&@I-����@a�46;�W(?i�g�xu�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_30(1-����&@9-����&@A-����&@I-����&@a�46;�W(?i�����?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_9(1-����&@9-����&@A-����&@I-����&@a�46;�W(?iI�Iu��?�Unknown
{�HostMul")gradient_tape/model/gp_layer/mul_20/Mul_1(1-����&@9-����&@A-����&@I-����&@a�46;�W(?i�����?�Unknown
m�HostSquare"model/gp_layer/Square_10(1-����&@9-����&@A-����&@I-����&@a�46;�W(?i5r�	�?�Unknown
g�HostMul"model/gp_layer/mul_24(1-����&@9-����&@A-����&@I-����&@a�46;�W(?ir�t��?�Unknown
g�HostMul"model/gp_layer/mul_25(1-����&@9-����&@A-����&@I-����&@a�46;�W(?i՛�n��?�Unknown
n�HostRealDiv"model/gp_layer/truediv_5(1-����&@9-����&@A-����&@I-����&@a�46;�W(?i8O<��?�Unknown
i�HostLess"Adam/gradients/Less_15(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a��l�'?i9���?�Unknown
i�HostAddV2"model/gp_layer/add_16(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a��l�'?i:�	��?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_4(1+�Y&@9+�Y@A+�Y&@I+�Y@a��l�'?i;�p��?�Unknown
�HostSelectV2"(model/gp_layer/softplus/forward/SelectV2(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a��l�'?i<{���?�Unknown
�HostSoftplus"(model/gp_layer/softplus/forward/Softplus(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a��l�'?i=F>��?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_21(1j�t�X&@9j�t�X@Aj�t�X&@Ij�t�X@a5� ��'?i�Q���?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_20/Mul(1j�t�X&@9j�t�X&@Aj�t�X&@Ij�t�X&@a5� ��'?i�\���?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write_1/TensorListSetItem_grad/TensorListSetItem(1
ףp=�%@9
ףp=�@A
ףp=�%@I
ףp=�@a	�)�F('?i}?j��?�Unknown
��HostNeg"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/Neg(1
ףp=�%@9
ףp=�@A
ףp=�%@I
ףp=�@a	�)�F('?i"v�d�?�Unknown
x�HostSum"&gradient_tape/model/gp_layer/add_5/Sum(1
ףp=�%@9
ףp=�%@A
ףp=�%@I
ףp=�%@a	�)�F('?i��r��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_19/Mul(1
ףp=�%@9
ףp=�%@A
ףp=�%@I
ףp=�%@a	�)�F('?iW�I�I�?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1}?5^��%@9}?5^��%@A}?5^��%@I}?5^��%@aP�.R�''?iF
�r��?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_2/TensorListGetItem_grad/TensorListReserve(1}?5^��%@9}?5^��@A}?5^��%@I}?5^��@aP�.R�''?i5-��.!�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_21_grad/TensorListPopBack(1}?5^��%@9}?5^��@A}?5^��%@I}?5^��@aP�.R�''?i$Pij�"�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_13(1}?5^��%@9}?5^��%@A}?5^��%@I}?5^��%@aP�.R�''?is�$�?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_22/Sum(1}?5^��%@9}?5^��%@A}?5^��%@I}?5^��%@aP�.R�''?i��a�%�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_8(1}?5^��%@9}?5^��@A}?5^��%@I}?5^��@aP�.R�''?i���&�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_4_grad/Shape/TensorListPopBack(1�Zd;%@9�Zd;@A�Zd;%@I�Zd;@a$���v�&?i-���a(�?�Unknown
z�HostNeg"(gradient_tape/model/gp_layer/truediv/Neg(1�Zd;%@9�Zd;%@A�Zd;%@I�Zd;%@a$���v�&?ii�b��)�?�Unknown
g�HostPack"model/gp_layer/stack(1�Zd;%@9�Zd;%@A�Zd;%@I�Zd;%@a$���v�&?i����3+�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_19(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@ak����&?i1��,�?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@ak����&?i�@�.�?�Unknown
��HostMatrixBandPart"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/MatrixBandPart(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@ak����&?iIW��n/�?�Unknown
p�HostSelectV2"Adam/gradients/SelectV2_1(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?iՑ���0�?�Unknown
��HostTensorListFromTensor"Hgradient_tape/model/gp_layer/map/TensorArrayV2Stack/TensorListFromTensor(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?ia�h�@2�?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_23/Sum(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?i�!�3�?�Unknown
t�HostMul""gradient_tape/model/gp_layer/mul_7(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?iyA��5�?�Unknown
n�HostMaximum"model/gp_layer/Maximum_2(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?i|��{6�?�Unknown
l�HostSquare"model/gp_layer/Square_1(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?i��I��7�?�Unknown
f�HostTile"model/gp_layer/Tile(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?i��M9�?�Unknown
��HostEqual"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes/Equal(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@ak����&?i�+��:�?�Unknown
y�HostStridedSlice"model/gp_layer/strided_slice_9(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?i5fr�<�?�Unknown
��HostSelectV2"2model/gp_layer/truediv_3/softplus/forward/SelectV2(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@ak����&?i��*�=�?�Unknown
��Host	ZerosLike"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write_1/TensorListSetItem_grad/zeros_like(1y�&1�$@9y�&1�@Ay�&1�$@Iy�&1�@a��"��%?i���d�>�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_10_grad/Shape/TensorListPopBack(1y�&1�$@9y�&1�@Ay�&1�$@Iy�&1�@a��"��%?iE��G@�?�Unknown
��HostStridedSlice"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/strided_slice_1(1y�&1�$@9y�&1�@Ay�&1�$@Iy�&1�@a��"��%?i<�\h�A�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_15/Mul(1y�&1�$@9y�&1�$@Ay�&1�$@Iy�&1�$@a��"��%?ie��C�?�Unknown
��HostAddN">model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_add_14(1y�&1�$@9y�&1�$@Ay�&1�$@Iy�&1�$@a��"��%?i�;�kfD�?�Unknown
h�HostAddN"Adam/gradients/AddN_1(1�Q���$@9�Q���$@A�Q���$@I�Q���$@a͝'i��%?i����E�?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem_grad/TensorListElementShape(1�Q���$@9�Q���@A�Q���$@I�Q���@a͝'i��%?i�`�]%G�?�Unknown
n�HostSigmoid"Adam/gradients/Sigmoid_8(1�Q���$@9�Q���$@A�Q���$@I�Q���$@a͝'i��%?i���քH�?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_3/RealDiv_1(1�Q���$@9�Q���$@A�Q���$@I�Q���$@a͝'i��%?iv��O�I�?�Unknown
f�HostExp"Adam/gradients/Exp_3(1w��/$@9w��/$@Aw��/$@Iw��/$@a�n��K`%?i=�T:K�?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_3(1w��/$@9w��/$@Aw��/$@Iw��/$@a�n��K`%?iYjY�L�?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_4(1w��/$@9w��/$@Aw��/$@Iw��/$@a�n��K`%?i��(^�M�?�Unknown
g�HostMul"model/gp_layer/mul_22(1w��/$@9w��/$@Aw��/$@Iw��/$@a�n��K`%?i�,�b<O�?�Unknown
��HostAddV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/add(1�&1�$@9�&1�@A�&1�$@I�&1�@a�t���_%?i���^�P�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/Sum(1�&1�$@9�&1�@A�&1�$@I�&1�@a�t���_%?i���Z�Q�?�Unknown
��HostTensorListFromTensor"Jgradient_tape/model/gp_layer/map/TensorArrayV2Stack_1/TensorListFromTensor(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a�t���_%?i�*W>S�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_24/Mul(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a�t���_%?i��S�T�?�Unknown
��HostMatrixBandPart"Tmodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/MatrixBandPart(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a�t���_%?iO�U�?�Unknown
f�HostSum"model/gp_layer/Sum_5(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a�t���_%?i)!K@W�?�Unknown
g�HostMul"model/gp_layer/mul_26(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a�t���_%?i3�*G�X�?�Unknown
��HostMatrixDiagPartV3"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/diag_part(1\���($@9\���(@A\���($@I\���(@a.{�N5_%?i��:�Y�?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_1/TensorListGetItem_grad/TensorListReserve(1u�V�#@9u�V�@Au�V�#@Iu�V�@a�E|�$?i�>A�8[�?�Unknown
h�HostAddN"Adam/gradients/AddN_3(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i� NA�\�?�Unknown
i�HostLess"Adam/gradients/Less_12(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?ii�Z��]�?�Unknown
��HostAddN"/Adam/gradients/PartitionedCall/gradients/AddN_3(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i�g?_�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_11(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@aL���$?i�Et�j`�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_11(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i��=�a�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_17(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i=ɍ�c�?�Unknown
��HostDynamicStitch"*gradient_tape/model/gp_layer/DynamicStitch(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i�;Pd�?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_3/TensorListFromTensor(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i�L���e�?�Unknown
��HostBroadcastTo"zmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/BroadcastTo_1(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@aL���$?i\�9�f�?�Unknown
��HostTensorListPushBack"model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@aL���$?i���5h�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_21(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@aL���$?iƑ�7�i�?�Unknown
��HostPack"~model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_5/values_1(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@aL���$?i{Sڶ�j�?�Unknown
g�HostMul"model/gp_layer/mul_12(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i0�5l�?�Unknown
f�HostMul"model/gp_layer/mul_2(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i���gm�?�Unknown
�HostEqual"+model/likelihood_layer/assert_equal_1/Equal(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@aL���$?i�� 4�n�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/add_grad/Sum_1(1Zd;�O�#@9Zd;�O�@AZd;�O�#@IZd;�O�@aIR �e�$?i��X� p�?�Unknown
u�HostReadVariableOp"Adam/Cast/ReadVariableOp(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?i�sh�Cq�?�Unknown
i�HostLess"Adam/gradients/Less_13(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?iCMx��r�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_3(1��"���"@9��"���@A��"���"@I��"���@a#�� 0$?i�&���s�?�Unknown
��HostGreaterEqual"+gradient_tape/model/gp_layer/GreaterEqual_1(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?i����u�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_6(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?i9٧�Ov�?�Unknown
��HostTensorListStack"5model/gp_layer/map/TensorArrayV2Stack/TensorListStack(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?i�����w�?�Unknown
g�HostMul"model/gp_layer/mul_21(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?i݋Ǹ�x�?�Unknown
y�HostStridedSlice"model/gp_layer/strided_slice_1(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?i/e׺z�?�Unknown
��HostSoftplus"2model/gp_layer/truediv_3/softplus/forward/Softplus(1��"���"@9��"���"@A��"���"@I��"���"@a#�� 0$?i�>�[{�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_4_grad/BroadcastGradientArgs/TensorListPopBack(1X9��v�"@9X9��v�@AX9��v�"@IX9��v�@ad)���/$?i$XB��|�?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem_grad/TensorListReserve(1X9��v�"@9X9��v�@AX9��v�"@IX9��v�@ad)���/$?i�q���}�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_4(1X9��v�"@9X9��v�"@AX9��v�"@IX9��v�"@ad)���/$?ij���$�?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_5(1X9��v�"@9X9��v�@AX9��v�"@IX9��v�@ad)���/$?i�S�g��?�Unknown
l�HostRealDiv"model/gp_layer/truediv(1X9��v�"@9X9��v�"@AX9��v�"@IX9��v�"@ad)���/$?i�������?�Unknown
��HostTensorListGetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem(1q=
ףp"@9q=
ףp@Aq=
ףp"@Iq=
ףp@a��	{ܘ#?iOov)��?�Unknown
��HostAddV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/add(1㥛� p"@9㥛� p@A㥛� p"@I㥛� p@a8�/Q�#?i?`����?�Unknown
d�HostMul"model/gp_layer/mul(1㥛� p"@9㥛� p"@A㥛� p"@I㥛� p"@a8�/Q�#?i/Q�3W��?�Unknown
��HostSelectV2"*model/gp_layer/softplus_5/forward/SelectV2(1㥛� p"@9㥛� p"@A㥛� p"@I㥛� p"@a8�/Q�#?iB�����?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_11_grad/Shape/TensorListPopBack(1V-��o"@9V-��o@AV-��o"@IV-��o@a �ŗ#?i_s5ʇ�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_1(1V-��o"@9V-��o"@AV-��o"@IV-��o"@a �ŗ#?i��k���?�Unknown
��Host	Transpose"0gradient_tape/model/gp_layer/transpose/transpose(1V-��o"@9V-��o"@AV-��o"@IV-��o"@a �ŗ#?i���-=��?�Unknown
��HostLess"smodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/cond/_126/model/gp_layer/map/while/Less_1(1�z�G�!@9ףp=
�@A�z�G�!@Iףp=
�@aSш`� #?il��5m��?�Unknown
i�HostAddN"Adam/gradients/AddN_25(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?iI'A5���?�Unknown
f�HostMul"Adam/gradients/mul_4(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i&p�4͍�?�Unknown
q�Host	ZerosLike"Adam/gradients/zeros_like(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i�4���?�Unknown
i�HostRandomShuffle"RandomShuffle(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i�e3-��?�Unknown
��Host
Reciprocal"Rgradient_tape/model/gp_layer/LogNormal_1/log_prob/LogNormal_exp/inverse/Reciprocal(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i�J�2]��?�Unknown
��HostSlice"Jgradient_tape/model/gp_layer/fill_triangular/forward/fill_triangular/Slice(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i��'2���?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_16/Mul(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?iw܈1���?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_23/Mul(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?iT%�0��?�Unknown
��HostAddN">model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_add_18(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i1nK0��?�Unknown
��HostAddV2"6model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_add_1(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i��/M��?�Unknown
h�HostSqrt"model/gp_layer/Sqrt_1(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i��/}��?�Unknown
f�HostSum"model/gp_layer/Sum_4(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i�Ho.���?�Unknown
��HostTensorListSetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Write/TensorListSetItem(1T㥛��!@9T㥛��@AT㥛��!@IT㥛��@a�׍��"?i���-ݚ�?�Unknown
��HostReadVariableOp".model/gp_layer/softplus/forward/ReadVariableOp(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i��1-��?�Unknown
��HostReadVariableOp"0model/gp_layer/softplus_6/forward/ReadVariableOp(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i_#�,=��?�Unknown
��HostSoftplus"2model/gp_layer/truediv_4/softplus/forward/Softplus(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i<l�+m��?�Unknown
��HostStridedSlice"&model/likelihood_layer/strided_slice_2(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�׍��"?i�U+���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_13_grad/Shape/TensorListPopBack(1�K7�A�!@9�K7�A�@A�K7�A�!@I�K7�A�@a�ݒ�j�"?iG>"͠�?�Unknown
i�HostAddN"Adam/gradients/AddN_24(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a��F&h"?if���?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem(1R���Q!@9R���Q@AR���Q!@IR���Q@a��F&h"?i=��&��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_33(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a��F&h"?i�_/�@��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_22/Mul(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a��F&h"?i3��+g��?�Unknown
x�HostMul"&gradient_tape/model/gp_layer/mul_5/Mul(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a��F&h"?i� �����?�Unknown
~�HostSum",gradient_tape/model/gp_layer/truediv_2/Sum_1(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a��F&h"?i)�\0���?�Unknown
��HostStridedSlice"Dmodel/gp_layer/fill_triangular/forward/fill_triangular/strided_slice(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a��F&h"?i����ڨ�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_22(1R���Q!@9R���Q@AR���Q!@IR���Q@a��F&h"?iB%5��?�Unknown
i�HostAddN"Adam/gradients/AddN_10(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a����g"?i��Ԯ'��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_19(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a����g"?i���(N��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_18/Mul(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a����g"?i�$4�t��?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_2/RealDiv_2(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a����g"?iK�����?�Unknown
��HostEqual"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/Equal(1� �rhQ!@9� �rhQ@A� �rhQ!@I� �rhQ@a����g"?if�����?�Unknown
g�HostMul"model/gp_layer/mul_19(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a����g"?i�C��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_22(1P��n� @9P��n� @AP��n� @IP��n� @aЅ�wV�!?i�~���?�Unknown
_�HostGatherV2"GatherV2(1P��n� @9P��n� @AP��n� @IP��n� @aЅ�wV�!?i�"��?�Unknown
��HostAddV2"Cmodel/gp_layer/LogNormal_2/log_prob/LogNormal_Normal_1/log_prob/add(1P��n� @9P��n� @AP��n� @IP��n� @aЅ�wV�!?i)oy?��?�Unknown
g�HostSum"model/gp_layer/Sum_11(1P��n� @9P��n� @AP��n� @IP��n� @aЅ�wV�!?iA��$\��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_11(1P��n� @9P��n�@AP��n� @IP��n�@aЅ�wV�!?iY_H*y��?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_1(1��(\�� @9��(\��@A��(\�� @I��(\��@a��+��!?i��&���?�Unknown
��HostTensorListLength"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_1/TensorListGetItem_grad/TensorListLength(1��(\�� @9��(\��@A��(\�� @I��(\��@a��+��!?i+Э#���?�Unknown
g�HostMul"Adam/gradients/mul_13(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?i��` й�?�Unknown
��HostDynamicStitch",gradient_tape/model/gp_layer/DynamicStitch_1(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?i�@��?�Unknown
{�HostSum")gradient_tape/model/gp_layer/add_16/Sum_1(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?if��
��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_21/Mul(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?iϱx'��?�Unknown
t�HostMul""gradient_tape/model/gp_layer/mul_9(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?i8j+D��?�Unknown
��HostRealDiv"Gmodel/gp_layer/LogNormal_2/log_prob/LogNormal_Normal_1/log_prob/truediv(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?i�"�a��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_6(1��(\�� @9��(\��@A��(\�� @I��(\��@a��+��!?i
ې~��?�Unknown
��HostSub"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/sub_3(1��(\�� @9��(\��@A��(\�� @I��(\��@a��+��!?is�C	���?�Unknown
w�HostStridedSlice"model/gp_layer/strided_slice(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?i�K����?�Unknown
n�HostRealDiv"model/gp_layer/truediv_2(1��(\�� @9��(\�� @A��(\�� @I��(\�� @a��+��!?iE����?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_12_grad/TensorListPopBack(1NbX94 @9NbX94@ANbX94 @INbX94@a�\���8!?i������?�Unknown
^�HostCast"Adam/Cast_3(1�ʡE�3 @9�ʡE�3 @A�ʡE�3 @I�ʡE�3 @a2c ]�7!?id�
���?�Unknown
f�HostExp"Adam/gradients/Exp_9(1�ʡE�3 @9�ʡE�3 @A�ʡE�3 @I�ʡE�3 @a2c ]�7!?i4���?�Unknown*��
wHostExp"&model/likelihood_layer/exp/forward/Exp(1^�I�|@9^�I�|@A^�I�|@I^�I�|@a�F����?i�F����?�Unknown
�Host	ReverseV2"@model/gp_layer/fill_triangular/forward/fill_triangular/ReverseV2(1��v���{@9��v���{@A��v���{@I��v���{@a��w@o��?i�[�v,�?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1sh��|��@9sh��|��@Aq=
ף{@Iq=
ף{@a�� ���?i����>Z�?�Unknown
�HostStridedSliceGrad"Egradient_tape/model/likelihood_layer/strided_slice_3/StridedSliceGrad(1�z�G�v@9�z�G�v@A�z�G�v@I�z�G�v@a`|6�5�?i�s_��?�Unknown
�Host
LogicalAnd"wmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/cond/_126/model/gp_layer/map/while/LogicalAnd(1;�O��Bv@9�)g�]@A;�O��Bv@I�)g�]@a\Ķ���?i�t�����?�Unknown
kHostMul"model/likelihood_layer/mul(1F���Բr@9F���Բr@AF���Բr@IF���Բr@a�=�� L�?i.b�b|
�?�Unknown
�HostStridedSliceGrad"Egradient_tape/model/likelihood_layer/strided_slice_4/StridedSliceGrad(1��Mb�q@9��Mb�q@A��Mb�q@I��Mb�q@a�ar�s`�?ii�`֊�?�Unknown
�HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/sub_3_grad/Sum_1(1j�t�hq@9j�t�ha@Aj�t�hq@Ij�t�ha@a��F#-�?ib��z��?�Unknown
g	HostLess"Adam/gradients/Less_4(1���S�m@9���S�m@A���S�m@I���S�m@aoA��[�?iS�M���?�Unknown
�
HostBatchMatMulV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_1_grad/MatMul(1�Vl@9�V\@A�Vl@I�V\@a�����?i��Yan�?�Unknown
}HostSum",gradient_tape/model/likelihood_layer/add/Sum(1�A`��nj@9�A`��nj@A�A`��nj@I�A`��nj@a��Fͽs�?inP.>I��?�Unknown
�HostBatchMatMulV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_1_grad/MatMul_1(1y�&1Tj@9y�&1TZ@Ay�&1Tj@Iy�&1TZ@a����Z�?i�����z�?�Unknown
�HostMatrixTriangularSolve"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve(1���S�i@9���S�Y@A���S�i@I���S�Y@a���& ��?i�$�H��?�Unknown
�HostCholesky"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Cholesky(1��~j��a@9��~j��Q@A��~j��a@I��~j��Q@a���Ɖ�?ib\���?�Unknown
�HostBatchMatMulV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/MatMul_1(1V-��ca@9V-��cQ@AV-��ca@IV-��cQ@a�%&3�?i��B�E��?�Unknown
�HostMatrixTriangularSolve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/triangular_solve/MatrixTriangularSolve(1!�rh�E`@9!�rh�EP@A!�rh�E`@I!�rh�EP@af#���~?i�g2R��?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1����̸h@9����̸h@A�G�z�^@I�G�z�^@a,��|?iz`�Ej�?�Unknown
}HostMul",gradient_tape/model/likelihood_layer/mul/Mul(1w��/=^@9w��/=^@Aw��/=^@Iw��/=^@ak=�{?i&qTF*��?�Unknown
mHostAddV2"model/likelihood_layer/add(1R����]@9R����]@AR����]@IR����]@a	Zޕ{?i�����H�?�Unknown
�HostLog"Bmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/Log(1V-���\@9V-���\@AV-���\@IV-���\@a���:R�z?i1�;��?�Unknown
�HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/MatMul(1#��~j<[@9#��~j<K@A#��~j<[@I#��~j<K@a�2&G�1y?i�����?�Unknown
eHostSum"model/gp_layer/Sum_1(1��(\�*[@9��(\�*[@A��(\�*[@I��(\�*[@a����j!y?itXp�|�?�Unknown
�HostSign"Cmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/Sign(1L7�A`�Z@9L7�A`�Z@AL7�A`�Z@IL7�A`�Z@a�J��x?i����>��?�Unknown
�HostMatrixTriangularSolve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/triangular_solve/MatrixTriangularSolve(1fffffZ@9fffffJ@AfffffZ@IfffffJ@a����Y)x?i	��L�?�?�Unknown
�HostSub"Bmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub(1
ףp=�W@9
ףp=�W@A
ףp=�W@I
ףp=�W@a�:�v?i���D��?�Unknown
�HostBroadcastTo"0gradient_tape/model/likelihood_layer/BroadcastTo(1�S㥛�W@9�S㥛�W@A�S㥛�W@I�S㥛�W@a�t�u?i=~�`���?�Unknown
�HostMatrixBandPart"{model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/MatrixBandPart(1m����:U@9m����:E@Am����:U@Im����:E@a:<���s?i%om>�?�Unknown
�HostRealDiv"^gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/RealDiv_1(11�ZdT@91�ZdT@A1�ZdT@I1�ZdT@a."�)�r?i��W���?�Unknown
�HostRealDiv"\gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/RealDiv(1���S�]R@9���S�]R@A���S�]R@I���S�]R@a�!L���p?i6Q����?�Unknown
YHostPow"Adam/Pow(1�K7�A(R@9�K7�A(R@A�K7�A(R@I�K7�A(R@a������p?i���?�Unknown
}HostSum",gradient_tape/model/likelihood_layer/mul/Sum(1�V�Q@9�V�Q@A�V�Q@I�V�Q@a�!��hp?i��ʿ�R�?�Unknown
� HostRealDiv"`gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1/RealDiv_2(1�E����Q@9�E����Q@A�E����Q@I�E����Q@a婳�hp?iM�q�M��?�Unknown
�!HostSquare"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Square_1(1�E����Q@9�E����A@A�E����Q@I�E����A@a婳�hp?i��o���?�Unknown
w"HostSum"&gradient_tape/model/gp_layer/mul_3/Sum(1}?5^�QQ@9}?5^�QQ@A}?5^�QQ@I}?5^�QQ@a�Y�|p?i\3_�?�Unknown
�#HostNeg"Xgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/Neg(1Zd;�?Q@9Zd;�?Q@AZd;�?Q@IZd;�?Q@a��A���o?i���<�U�?�Unknown
�$HostRealDiv"`gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1/RealDiv_1(1���Mb�P@9���Mb�P@A���Mb�P@I���Mb�P@aQ#,ȫeo?i��{����?�Unknown
X%HostSlice"Slice(1��C�lWP@9��C�lWP@A��C�lWP@I��C�lWP@a�C���;n?iQ�T��?�Unknown
�&HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatMul_1(1!�rh�EP@9!�rh�E@@A!�rh�EP@I!�rh�E@@af#���n?i��US�?�Unknown
'HostMul".gradient_tape/model/likelihood_layer/mul_1/Mul(1w��/mO@9w��/mO@Aw��/mO@Iw��/mO@a�B�uPm?ifA�wG�?�Unknown
�(HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1��"��NN@9��"��NN@A��"��NN@I��"��NN@a]����	l?i%�����?�Unknown
e)HostSum"model/gp_layer/Sum_6(1%��C+N@9%��C+N@A%��C+N@I%��C+N@a7c�|��k?i�ѕ�[��?�Unknown
�*HostStridedSlice"&model/likelihood_layer/strided_slice_4(1�ʡE��M@9�ʡE��M@A�ʡE��M@I�ʡE��M@aP3��S�k?iR�*n���?�Unknown
�+HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatMul(1w��/M@9w��/=@Aw��/M@Iw��/=@a3�l��j?ix�3.h$�?�Unknown
�,HostStridedSlice"&model/likelihood_layer/strided_slice_3(1w��/M@9w��/M@Aw��/M@Iw��/M@a3�l��j?i��<�'Z�?�Unknown
�-HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatMul_2(1�(\���L@9�(\���<@A�(\���L@I�(\���<@a�Rb�͝j?iDc��c��?�Unknown
e.HostSum"model/gp_layer/Sum_3(1H�z��K@9H�z��K@AH�z��K@IH�z��K@a�6*)�i?iJ��ۍ��?�Unknown
�/HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_2(1�p=
׃K@9�p=
׃;@A�p=
׃K@I�p=
׃;@a!s0�ti?i01(�u��?�Unknown
�0HostNeg"Tgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub/Neg(1��x�&�J@9��x�&�J@A��x�&�J@I��x�&�J@a��d��h?i�\�I'�?�Unknown
�1HostSelectV2"*model/gp_layer/LogNormal/log_prob/SelectV2(1}?5^��J@9}?5^��J@A}?5^��J@I}?5^��J@aw���h?i;t�,X�?�Unknown
{2HostMatrixDiagV3"!gradient_tape/model/gp_layer/diag(1��v���I@9��v���I@A��v���I@I��v���I@a�B��Ah?i�^0=��?�Unknown
�3HostRealDiv"^gradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/RealDiv_2(1���(\�I@9���(\�I@A���(\�I@I���(\�I@a����g?i�(�"���?�Unknown
y4HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1bX9��H@9bX9��H@AbX9��H@IbX9��H@a�bäu�f?ik��D��?�Unknown
�5HostSoftplus"2model/gp_layer/truediv_2/softplus/forward/Softplus(1bX9��H@9bX9��H@AbX9��H@IbX9��H@a�bäu�f?i169� �?�Unknown
�6HostNeg"Vgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub_3/Neg(1!�rh�MH@9!�rh�MH@A!�rh�MH@I!�rh�MH@ao�Z{f?i6�\��?�?�Unknown
s7HostMul""gradient_tape/model/gp_layer/mul_2(1=
ףpMH@9=
ףpMH@A=
ףpMH@I=
ףpMH@a���;{f?i\�$�l�?�Unknown
�8HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_8_grad/TensorListPopBack(1}?5^�)H@9}?5^�)8@A}?5^�)H@I}?5^�)8@a��2Zf?iA`����?�Unknown
�9HostRealDiv"Hmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1(1;�O���G@9;�O���G@A;�O���G@I;�O���G@a9����e?ie������?�Unknown
m:HostMul"model/likelihood_layer/mul_1(1���x�vG@9���x�vG@A���x�vG@I���x�vG@a�r��Ǵe?iJ��G���?�Unknown
c;HostExp"model/gp_layer/Exp(133333SG@933333SG@A33333SG@I33333SG@a�R�ʾ�e?i�܈�!�?�Unknown
�<HostSub"Dmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub_3(1sh��|/G@9sh��|/G@Ash��|/G@Ish��|/G@ak2���re?iT�0G�?�Unknown
g=HostAddN"Adam/gradients/AddN_5(1��n��F@9��n��F@A��n��F@I��n��F@a5�r�r�d?i����p�?�Unknown
�>HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_23(1�����|F@9�����|6@A�����|F@I�����|6@a�mTi�d?i�o�~��?�Unknown
�?HostMul"Vgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul_1/Mul(1�&1�|F@9�&1�|F@A�&1�|F@I�&1�|F@au�lK�d?i��x~��?�Unknown
}@HostSelectV2"'gradient_tape/model/gp_layer/SelectV2_3(1h��|?5F@9h��|?5F@Ah��|?5F@Ih��|?5F@a'rb�8�d?ieL��/��?�Unknown
�AHostAddN"/Adam/gradients/PartitionedCall/gradients/AddN_4(1� �rhF@9� �rhF@A� �rhF@I� �rhF@ahb\!jd?i* �?�Unknown
sBHost	ZerosLike"Adam/gradients/zeros_like_26(1� �rhF@9� �rhF@A� �rhF@I� �rhF@ahb\!jd?i�B4�>�?�Unknown
�CHostMul"Bmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul(1� �rhF@9� �rhF@A� �rhF@I� �rhF@ahb\!jd?i�v�V�g�?�Unknown
�DHostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1�Q���R@9�Q���R@A����M�E@I����M�E@a�F���c?i� �w��?�Unknown
�EHostMul"Pgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul(1�I+E@9�I+E@A�I+E@I�I+E@a#�6*��c?i�oT�|��?�Unknown
�FHostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/strided_slice_1(1�I+E@9�I+5@A�I+E@I�I+5@a#�6*��c?iAݨ���?�Unknown
�GHostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_grad/MatMul(1�~j�t�D@9�~j�t�4@A�~j�t�D@I�~j�t�4@a��1��ac?iE@�E�?�Unknown
�HHostMul"Zgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv_1/mul(1�~j�t�D@9�~j�t�D@A�~j�t�D@I�~j�t�D@a��1��ac?iI��2+�?�Unknown
hIHostAddV2"model/gp_layer/add_10(1q=
ף@D@9q=
ף@D@Aq=
ף@D@Iq=
ף@D@ad �b?iM�Us�P�?�Unknown
}JHostSelectV2"'gradient_tape/model/gp_layer/SelectV2_2(1�rh��D@9�rh��D@A�rh��D@I�rh��D@a��3�b?i񻡶u�?�Unknown
�KHostMul"Xgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv/mul(1�����D@9�����D@A�����D@I�����D@aG����b?i�����?�Unknown
�LHostNeg"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/sub_3_grad/Neg(1L7�A`�C@9L7�A`�3@AL7�A`�C@IL7�A`�3@a����Xb?iZ�`���?�Unknown
�MHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_6_grad/Shape/TensorListPopBack(1�l����C@9�l����3@A�l����C@I�l����3@aґ P�7b?i~$��?�Unknown
�NHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate(1�O��n�E@9�O��n�E@A�l����C@I�l����C@aґ P�7b?i���}�?�Unknown
�OHostRealDiv"Fmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/truediv(1'1�jC@9'1�jC@A'1�jC@I'1�jC@a�a����a?if	 /i,�?�Unknown
~PHostMatMul"*gradient_tape/model/gp_layer/MatMul/MatMul(1D�l��iC@9D�l��iC@AD�l��iC@ID�l��iC@aRr�M��a?iK�KTP�?�Unknown
uQHostFlushSummaryWriter"FlushSummaryWriter(1��(\�"C@9��(\�"C@A��(\�"C@I��(\�"C@a2��{�a?i��sC�s�?�Unknown�
�RHostMul"4gradient_tape/model/likelihood_layer/exp/forward/mul(1��(\�"C@9��(\�"C@A��(\�"C@I��(\�"C@a2��{�a?i�K;"��?�Unknown
�SHost	ZerosLike"Wgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/zeros_like(1��(\�"C@9��(\�"C@A��(\�"C@I��(\�"C@a2��{�a?iwo#3���?�Unknown
�THostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_7_grad/TensorListPopBack(1���K�B@9���K�2@A���K�B@I���K�2@a���BPa?i;#5�)��?�Unknown
kUHostMatMul"model/gp_layer/MatMul_1(19��v�oB@99��v�oB@A9��v�oB@I9��v�oB@a��Va?i����E��?�Unknown
}VHostSelectV2"'gradient_tape/model/gp_layer/SelectV2_1(1y�&1LB@9y�&1LB@Ay�&1LB@Iy�&1LB@a��%�`?i�S.�!�?�Unknown
�WHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/triangular_solve/MatrixTriangularSolve/TensorListPopBack(1�G�zB@9�G�z2@A�G�zB@I�G�z2@a�a�sת`?i���uB�?�Unknown
�XHostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_13(1T㥛��A@9T㥛��1@AT㥛��A@IT㥛��1@a�A�BΉ`?iC�5�c�?�Unknown
gYHostAddN"Adam/gradients/AddN_8(1q=
ף�A@9q=
ף�A@Aq=
ף�A@Iq=
ף�A@a?R��`?i�������?�Unknown
gZHostAddN"Adam/gradients/AddN_7(1�rh��A@9�rh��A@A�rh��A@I�rh��A@a2���h`?i�m��?�Unknown
�[HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_grad/Mul(1�rh��A@9�rh��1@A�rh��A@I�rh��1@a2���h`?iw��0?��?�Unknown
�\HostSquare"smodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Square(1�rh��A@9�rh��1@A�rh��A@I�rh��1@a2���h`?i��
~��?�Unknown
�]HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_15(1�MbX�A@9�MbX�1@A�MbX�A@I�MbX�1@a���G`?io"����?�Unknown
�^HostBroadcastTo"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_1_grad/BroadcastTo(1+��A@9+��1@A+��A@I+��1@aW"�@G`?i��&z�?�Unknown
�_HostStridedSliceGrad"cgradient_tape/model/gp_layer/fill_triangular/forward/fill_triangular/strided_slice/StridedSliceGrad(1�l���QA@9�l���QA@A�l���QA@I�l���QA@a
��l`?isr�$�?�Unknown
�`HostMatMul"smodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/MatMul(1�l���QA@9�l���Q1@A�l���QA@I�l���Q1@a
��l`?iU�S"4�?�Unknown
�aHostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem_grad/TensorListReserve(1ˡE��-A@9ˡE��-1@AˡE��-A@IˡE��-1@aƃ;[��_?i���D�?�Unknown
�bHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_8_grad/Shape/TensorListPopBack(1ˡE��-A@9ˡE��-1@AˡE��-A@IˡE��-1@aƃ;[��_?i�P?�S�?�Unknown
scHost	ZerosLike"Adam/gradients/zeros_like_28(1ˡE��-A@9ˡE��-A@AˡE��-A@IˡE��-A@aƃ;[��_?i���~�c�?�Unknown
�dHostMatMul".gradient_tape/model/gp_layer/MatMul_1/MatMul_1(1��� ��@@9��� ��@@A��� ��@@I��� ��@@a���S_?i|��Ps�?�Unknown
zeHostMul")gradient_tape/model/gp_layer/mul_26/Mul_1(1��� ��@@9��� ��@@A��� ��@@I��� ��@@a���S_?i	��т�?�Unknown
�fHostMatMul"Mmodel/gp_layer/Tensordot_1/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1��"���@@9��"���@@A��"���@@I��"���@@aZ�3A�^?iё�1��?�Unknown
fgHostSum"model/gp_layer/Sum_12(1B`��"{@@9B`��"{@@AB`��"{@@IB`��"{@@aۃ1�}^?i4�p��?�Unknown
�hHostMatMul",gradient_tape/model/gp_layer/MatMul_1/MatMul(1�ʡE�3@@9�ʡE�3@@A�ʡE�3@@I�ʡE�3@@a?�l��]?i���m��?�Unknown
eiHostMul"model/gp_layer/mul_4(1�ʡE�3@@9�ʡE�3@@A�ʡE�3@@I�ʡE�3@@a?�l��]?i��j��?�Unknown
�jHost
Reciprocal"Wgradient_tape/model/likelihood_layer/model_likelihood_layer_Laplace/log_prob/Reciprocal(1�$��3@@9�$��3@@A�$��3@@I�$��3@@a$�̐�]?i){�g��?�Unknown
�kHostTensorListLength"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListLength(1���Q�?@9���Q�/@A���Q�?@I���Q�/@ap��lu]?i��8"��?�Unknown
dlHostDataset"Iterator::Model(1�MbXIO@9�MbXIO@A���Q�?@I���Q�?@ap��lu]?i�T����?�Unknown
hmHostPack"model/gp_layer/stack_2(1���Q�?@9���Q�?@A���Q�?@I���Q�?@ap��lu]?i������?�Unknown
�nHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1����xI?@9����xI/@A����xI?@I����xI/@a�"�DG�\?i0$�G	�?�Unknown
soHostDataset"Iterator::Model::ParallelMapV2(1�|?5^�>@9�|?5^�>@A�|?5^�>@I�|?5^�>@aî��l\?i�{��F�?�Unknown
rpHostTensorSliceDataset"TensorSliceDataset(1�|?5^�>@9�|?5^�>@A�|?5^�>@I�|?5^�>@aî��l\?i�җ-}%�?�Unknown
�qHostMul"Dmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/mul_1(1'1��>@9'1��>@A'1��>@I'1��>@a��@�l\?if)8��3�?�Unknown
�rHostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_3/TensorListStack(1��� �r>@9��� �r>@A��� �r>@I��� �r>@a���ޖ*\?i�z���A�?�Unknown
�sHostSum"pmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Sum(1��� �r>@9��� �r.@A��� �r>@I��� �r.@a���ޖ*\?i
��O�?�Unknown
�tHostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_grad/MatMul/TensorListPopBack(1j�t��=@9j�t��-@Aj�t��=@Ij�t��-@a����[?itp�]�?�Unknown
�uHostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem_grad/TensorListSetItem(1�p=
��=@9�p=
��-@A�p=
��=@I�p=
��-@a�"�r�[?i[���k�?�Unknown
svHostMul""gradient_tape/model/gp_layer/Mul_5(1�p=
��=@9�p=
��=@A�p=
��=@I�p=
��=@a�"�r�[?i-���Wy�?�Unknown
{wHostSelectV2"%gradient_tape/model/gp_layer/SelectV2(1�p=
��=@9�p=
��=@A�p=
��=@I�p=
��=@a�"�r�[?i>�+��?�Unknown
�xHostRealDiv"0gradient_tape/model/gp_layer/truediv_4/RealDiv_2(1\���(�=@9\���(�=@A\���(�=@I\���(�=@ah�#d[?i@*(-ݔ�?�Unknown
�yHost	Transpose"<gradient_tape/model/gp_layer/Tensordot_1/transpose/transpose(1��~j�T=@9��~j�T=@A��~j�T=@I��~j�T=@a�w�"[?i"f�5n��?�Unknown
�zHostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_5(1Zd;�O=@9Zd;�O-@AZd;�O=@IZd;�O-@âmT��Z?i㜭4ޯ�?�Unknown
�{HostStridedSlice"|model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/strided_slice_3(1X9��v~<@9X9��v~,@AX9��v~<@IX9��v~,@a0Y��[Z?id�u!��?�Unknown
u|HostTile"#gradient_tape/model/gp_layer/Tile_8(1��|?5~<@9��|?5~<@A��|?5~<@I��|?5~<@a�"W�[Z?i����9��?�Unknown
n}HostSigmoid"Adam/gradients/Sigmoid_12(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a��L��Z?if5�F��?�Unknown
�~HostSelectV2"<gradient_tape/model/gp_layer/LogNormal_2/log_prob/SelectV2_1(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a��L��Z?i�A|zS��?�Unknown
�HostRealDiv"0gradient_tape/model/gp_layer/truediv_1/RealDiv_2(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a��L��Z?iHh�?`��?�Unknown
��HostAddV2"Amodel/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Internal_0_add_1(1X9��6<@9X9��6<@AX9��6<@IX9��6<@a��L��Z?i��
m��?�Unknown
f�HostMul"Adam/gradients/mul_7(1J+�6<@9J+�6<@AJ+�6<@IJ+�6<@a}K�MZ?i;��y�?�Unknown
x�HostMul"&gradient_tape/model/gp_layer/mul_7/Mul(1���(\�;@9���(\�;@A���(\�;@I���(\�;@aa�B,x�Y?i��he�?�Unknown
��HostAddN"/Adam/gradients/PartitionedCall/gradients/AddN_1(1�v���;@9�v���;@A�v���;@I�v���;@a/�@�;�Y?i���Q%�?�Unknown
��HostStridedSliceGrad">gradient_tape/model/gp_layer/strided_slice_13/StridedSliceGrad(1H�z��;@9H�z��;@AH�z��;@IH�z��;@a�6*)�Y?i/s�2�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_3(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a�B,�SY?iP'�%�>�?�Unknown
��HostUnpack",gradient_tape/model/gp_layer/stack_2/unstack(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a�B,�SY?iq=;�nK�?�Unknown
g�HostMul"model/gp_layer/mul_10(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a�B,�SY?i�S�<X�?�Unknown
��HostStridedSlice"&model/likelihood_layer/strided_slice_8(1�K7�A`;@9�K7�A`;@A�K7�A`;@I�K7�A`;@a�B,�SY?i�i��d�?�Unknown
��HostSelectV2"*model/gp_layer/softplus_6/forward/SelectV2(1F����;@9F����;@AF����;@IF����;@aF"fY?i�z6JJq�?�Unknown
f�HostExp"model/gp_layer/Exp_3(1j�t�;@9j�t�;@Aj�t�;@Ij�t�;@a# ��Y?iƊ��}�?�Unknown
��HostAddV2"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/add_2(1j�t�;@9j�t�+@Aj�t�;@Ij�t�+@a# ��Y?iؚ�[��?�Unknown
��HostDynamicStitch"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_1_grad/DynamicStitch(1� �rh�:@9� �rh�*@A� �rh�:@I� �rh�*@a����X?i�����?�Unknown
��HostAddN"-Adam/gradients/PartitionedCall/gradients/AddN(1����x�:@9����x�:@A����x�:@I����x�:@aD�	bf�X?i��/���?�Unknown
l�HostConcatV2"model/gp_layer/concat(1����MB:@9����MB:@A����MB:@I����MB:@a)b��JX?iL�.��?�Unknown
��HostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_2/TensorListStack(15^�IB:@95^�IB:@A5^�IB:@I5^�IB:@a����SJX?i�0S��?�Unknown
f�HostExp"model/gp_layer/Exp_2(15^�IB:@95^�IB:@A5^�IB:@I5^�IB:@a����SJX?iΫZx��?�Unknown
��HostBroadcastTo"(gradient_tape/model/gp_layer/BroadcastTo(133333�9@933333�9@A33333�9@I33333�9@aZ�;/�W?iO�r[��?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_1(133333�9@933333�)@A33333�9@I33333�)@aZ�;/�W?iЖ��>��?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_22(1m����9@9m����)@Am����9@Im����)@a(#���W?ib�	�!��?�Unknown
g�HostMul"model/gp_layer/mul_17(1m����9@9m����9@Am����9@Im����9@a(#���W?i�W|��?�Unknown
��HostRealDiv".gradient_tape/model/gp_layer/truediv/RealDiv_1(1����k9@9����k9@A����k9@I����k9@a����W?iUpĊ��?�Unknown
n�HostSigmoid"Adam/gradients/Sigmoid_7(1�Q��k9@9�Q��k9@A�Q��k9@I�Q��k9@a���9��W?i�_�z��?�Unknown
��HostMatMul"Kmodel/gp_layer/Tensordot/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1j�t�$9@9j�t�$9@Aj�t�$9@Ij�t�$9@a�����AW?iJ�a)�?�Unknown
i�HostAddV2"model/gp_layer/add_22(1j�t�$9@9j�t�$9@Aj�t�$9@Ij�t�$9@a�����AW?ih4�H�%�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_14(1�&1��8@9�&1��(@A�&1��8@I�&1��(@a=b�u��V?i�t&J1�?�Unknown
��HostStridedSliceGrad"rgradient_tape/model/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/strided_slice/StridedSliceGrad(1�&1��8@9�&1��8@A�&1��8@I�&1��8@a=b�u��V?i��.�<�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/add_grad/Sum(1#��~j�8@9#��~j�(@A#��~j�8@I#��~j�(@a���~�V?i��IH�?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_1_grad/Mul_1(1h��|?�8@9h��|?�(@Ah��|?�8@Ih��|?�(@a�!���V?i�#��S�?�Unknown
v�HostSum"$gradient_tape/model/gp_layer/add/Sum(1�E����8@9�E����8@A�E����8@I�E����8@a�B�sl�V?i>�]N_�?�Unknown
i�HostLess"Adam/gradients/Less_11(1!�rh�M8@9!�rh�M8@A!�rh�M8@I!�rh�M8@ao�Z{V?i?|f�Dj�?�Unknown
��HostTensorListStack"7model/gp_layer/map/TensorArrayV2Stack_1/TensorListStack(1!�rh�M8@9!�rh�M8@A!�rh�M8@I!�rh�M8@ao�Z{V?i@Vo��u�?�Unknown
��HostBroadcastTo"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_grad/BroadcastTo(1Zd;�OM8@9Zd;�OM(@AZd;�OM8@IZd;�OM(@a<#�q{V?iR/(7���?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListSetItem(1Zd;�OM8@9Zd;�OM(@AZd;�OM8@IZd;�OM(@a<#�q{V?id�����?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_18(1���S�8@9���S�(@A���S�8@I���S�(@a��9V?iU�hK��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_grad/Shape/TensorListPopBack(1X9��v�7@9X9��v�'@AX9��v�7@IX9��v�'@a������U?i&�����?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_25/Mul(1ףp=
w7@9ףp=
w7@Aףp=
w7@Iףp=
w7@aRb�K�U?i�t�:��?�Unknown
l�HostSum"model/likelihood_layer/Sum(1ףp=
w7@9ףp=
w7@Aףp=
w7@Iףp=
w7@aRb�K�U?i�>�ʷ�?�Unknown
��HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate(1     @@9     @@AZd;�/7@IZd;�/7@a7��sU?i	P6���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/transpose_grad/InvertPermutation/TensorListPopBack(1V-��/7@9V-��/'@AV-��/7@IV-��/'@a"���rU?i��D�=��?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_18/Sum(1V-��/7@9V-��/7@AV-��/7@IV-��/7@a"���rU?i+�9
���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_22_grad/Shape/TensorListPopBack(1���(\/7@9���(\/'@A���(\/7@I���(\/'@a�B�I�rU?i�P�U���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem/TensorListPopBack(1�x�&1�6@9�x�&1�&@A�x�&1�6@I�x�&1�&@a��~��0U?i=��H��?�Unknown
^�HostCast"Adam/Cast_1(1-����6@9-����6@A-����6@I-����6@a�}�0U?i������?�Unknown
��HostTensorListGetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem(1-����6@9-����&@A-����6@I-����&@a�}�0U?i?��;y�?�Unknown
f�HostExp"Adam/gradients/Exp_1(1��n��6@9��n��6@A��n��6@I��n��6@a5�r�r�T?i�F�t��?�Unknown
��HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1��n��6@9��n��6@A��n��6@I��n��6@a5�r�r�T?i �g�?�Unknown
��HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1��n��6@9��n��&@A��n��6@I��n��&@a5�r�r�T?ib�Q��!�?�Unknown
��HostMatMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_grad/MatMul_1(1+�Y6@9+�Y&@A+�Y6@I+�Y&@a�h#`�T?i�mc5,�?�Unknown
��HostSelectV2"*model/gp_layer/softplus_7/forward/SelectV2(1F����X6@9F����X6@AF����X6@IF����X6@a��f�#�T?i� %)�6�?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1� �rh6@9� �rh6@A� �rh6@I� �rh6@ahb\!jT?i%ϵ1�@�?�Unknown
��HostSelectV2"2model/gp_layer/truediv_5/softplus/forward/SelectV2(1� �rh6@9� �rh6@A� �rh6@I� �rh6@ahb\!jT?iV}F:�J�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_4(1}?5^��5@9}?5^��%@A}?5^��5@I}?5^��%@a�BP�'T?iw%V	U�?�Unknown
n�HostSelectV2"Adam/gradients/SelectV2(1��(\��5@9��(\��5@A��(\��5@I��(\��5@a��G]��S?ihɄ�^�?�Unknown
��HostMatMul"bgradient_tape/model/gp_layer/Tensordot_1/MatMul/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1��(\��5@9��(\��5@A��(\��5@I��(\��5@a��G]��S?iYm��h�?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_1_grad/Mul(1����M�5@9����M�%@A����M�5@I����M�%@a�F���S?iZ���r�?�Unknown
��Host
Reciprocal"cgradient_tape/model/gp_layer/LogNormal/log_prob/LogNormal_exp_2/inverse_log_det_jacobian/Reciprocal(1����M�5@9����M�5@A����M�5@I����M�5@a�F���S?i[�p��|�?�Unknown
f�HostSum"model/gp_layer/Sum_8(1����M�5@9����M�5@A����M�5@I����M�5@a�F���S?i\VO�ǆ�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_13_grad/TensorListPopBack(1{�G�:5@9{�G�:%@A{�G�:5@I{�G�:%@aK�;[��S?i=��]���?�Unknown
��Host
Reciprocal"Pgradient_tape/model/gp_layer/LogNormal/log_prob/LogNormal_exp/inverse/Reciprocal(1{�G�:5@9{�G�:5@A{�G�:5@I{�G�:5@aK�;[��S?i��,k��?�Unknown
g�HostAdd"model/gp_layer/Add_15(1{�G�:5@9{�G�:5@A{�G�:5@I{�G�:5@aK�;[��S?i�/X�<��?�Unknown
f�HostMul"Adam/gradients/mul_1(1�~j�t�4@9�~j�t�4@A�~j�t�4@I�~j�t�4@a��1��aS?i������?�Unknown
��HostStridedSliceGrad">gradient_tape/model/gp_layer/strided_slice_14/StridedSliceGrad(1�~j�t�4@9�~j�t�4@A�~j�t�4@I�~j�t�4@a��1��aS?i�aQ����?�Unknown
{�HostAddV2"'model/gp_layer/LogNormal_3/log_prob/add(1�~j�t�4@9�~j�t�4@A�~j�t�4@I�~j�t�4@a��1��aS?iB��KO��?�Unknown
��HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1�/�$�E@9�/�$�E@A���ƫ4@I���ƫ4@a|b%�;S?i������?�Unknown
��HostRealDiv"]gradient_tape/model/gp_layer/LogNormal_2/log_prob/LogNormal_Normal_1/log_prob/truediv/RealDiv(1���ƫ4@9���ƫ4@A���ƫ4@I���ƫ4@a|b%�;S?i�Ňn��?�Unknown
o�HostSigmoid"Adam/gradients/Sigmoid_10(11�Zd4@91�Zd4@A1�Zd4@I1�Zd4@a."�)�R?i5�����?�Unknown
��HostMul"Sgradient_tape/model/gp_layer/LogNormal_1/log_prob/LogNormal_Normal_1/log_prob/mul_1(11�Zd4@91�Zd4@A1�Zd4@I1�Zd4@a."�)�R?i�:Z�K��?�Unknown
��HostSelectV2"<gradient_tape/model/gp_layer/LogNormal_3/log_prob/SelectV2_1(11�Zd4@91�Zd4@A1�Zd4@I1�Zd4@a."�)�R?iW�$F���?�Unknown
��HostAddV2"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/add_1(11�Zd4@91�Zd$@A1�Zd4@I1�Zd$@a."�)�R?i�U��(��?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_25(1�rh��4@9�rh��4@A�rh��4@I�rh��4@a��3�R?iYވfv�?�Unknown
��HostSelectV2":gradient_tape/model/gp_layer/LogNormal/log_prob/SelectV2_1(1/�$��3@9/�$��3@A/�$��3@I/�$��3@a���YR?i�a���?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/mul_1(1h��|?�3@9h��|?�#@Ah��|?�3@Ih��|?�#@a_�1�XR?i�	M��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_16(1�G�z�3@9�G�z�#@A�G�z�3@I�G�z�#@aDa�n�R?i<bA���?�Unknown
j�HostWriteSummary"WriteSummary(1����ҍ3@9����ҍ3@A����ҍ3@I����ҍ3@a��εR?i}�(!�'�?�Unknown�
|�HostNeg"*gradient_tape/model/gp_layer/truediv_2/Neg(1����ҍ3@9����ҍ3@A����ҍ3@I����ҍ3@a��εR?i�\|�0�?�Unknown
��HostAddN"=model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_add_8(1����ҍ3@9����ҍ3@A����ҍ3@I����ҍ3@a��εR?i�����9�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_9(1fffffF3@9fffffF#@AfffffF3@IfffffF#@a�A�l��Q?i R�(�B�?�Unknown
��Host	Transpose"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/adjoint/matrix_transpose/transpose(1fffffF3@9fffffF#@AfffffF3@IfffffF#@a�A�l��Q?iA�dz�K�?�Unknown
��HostConcatV2"Lmodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/concat(1fffffF3@9fffffF3@AfffffF3@IfffffF3@a�A�l��Q?ibB̻T�?�Unknown
��HostStridedSlice"Smodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/strided_slice(1fffffF3@9fffffF3@AfffffF3@IfffffF3@a�A�l��Q?i����]�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_13(1fffffF3@9fffffF#@AfffffF3@IfffffF#@a�A�l��Q?i�2�o�f�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_17(1��"���2@9��"���"@A��"���2@I��"���"@au�
��Q?i���Yo�?�Unknown
��HostDynamicStitch"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Sum_grad/DynamicStitch(1��"���2@9��"���"@A��"���2@I��"���"@au�
��Q?i�� #x�?�Unknown
p�HostSelectV2"Adam/gradients/SelectV2_2(1��Q��2@9��Q��2@A��Q��2@I��Q��2@aC"�jT�Q?i���*��?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_19/Sum(1���K�2@9���K�2@A���K�2@I���K�2@a���BPQ?i���K���?�Unknown
��HostTensorListFromTensor":model/gp_layer/map/TensorArrayUnstack/TensorListFromTensor(1Zd;�o2@9Zd;�o2@AZd;�o2@IZd;�o2@a��Ϧ/Q?iy_�c��?�Unknown
n�HostRealDiv"model/gp_layer/truediv_3(1Zd;�o2@9Zd;�o2@AZd;�o2@IZd;�o2@a��Ϧ/Q?iJ�s{���?�Unknown
��HostMatMul"`gradient_tape/model/gp_layer/Tensordot/MatMul/ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul(1V-��o2@9V-��o2@AV-��o2@IV-��o2@at���Q?i+.�t)��?�Unknown
x�HostMul"&gradient_tape/model/gp_layer/mul_1/Mul(1V-��o2@9V-��o2@AV-��o2@IV-��o2@at���Q?i�zn���?�Unknown
g�HostMul"model/gp_layer/mul_16(1�x�&1(2@9�x�&1(2@A�x�&1(2@I�x�&1(2@a&�ä��P?i���^��?�Unknown
�HostEqual"+model/likelihood_layer/assert_equal_3/Equal(1�x�&1(2@9�x�&1(2@A�x�&1(2@I�x�&1(2@a&�ä��P?i�XO|��?�Unknown
��HostStridedSlice"&model/likelihood_layer/strided_slice_7(1�x�&1(2@9�x�&1(2@A�x�&1(2@I�x�&1(2@a&�ä��P?iO�q?���?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_16(1T㥛��1@9T㥛��!@AT㥛��1@IT㥛��!@a�A�BΉP?i��&'��?�Unknown
��HostTensorListStack"Cgradient_tape/model/gp_layer/map/TensorArrayUnstack/TensorListStack(1T㥛��1@9T㥛��1@AT㥛��1@IT㥛��1@a�A�BΉP?i�s�l��?�Unknown
��HostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_1/TensorListStack(1T㥛��1@9T㥛��1@AT㥛��1@IT㥛��1@a�A�BΉP?i2������?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_17/Mul(1�MbX�1@9�MbX�1@A�MbX�1@I�MbX�1@a���GP?i�'�����?�Unknown
g�HostAdd"model/gp_layer/Add_11(1+��1@9+��1@A+��1@I+��1@aW"�@GP?iD~f����?�Unknown
��HostTensorListGetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem(1+��1@9+��!@A+��1@I+��!@aW"�@GP?i��R��?�Unknown
n�HostRealDiv"model/gp_layer/truediv_4(1+��1@9+��1@A+��1@I+��1@aW"�@GP?if+�@��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_14_grad/Shape/TensorListPopBack(1�l���Q1@9�l���Q!@A�l���Q1@I�l���Q!@a
��lP?i�|�B�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_grad/TensorListPopBack(1�l���Q1@9�l���Q!@A�l���Q1@I�l���Q!@a
��lP?iH΅~E�?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_4/TensorListFromTensor(1�l���Q1@9�l���Q1@A�l���Q1@I�l���Q1@a
��lP?i��4H�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_17_grad/Shape/TensorListPopBack(1
ףp=
1@9
ףp=
!@A
ףp=
1@I
ףp=
!@awC1���O?i
l3�)�?�Unknown
p�HostSelectV2"Adam/gradients/SelectV2_4(1
ףp=
1@9
ףp=
1@A
ףp=
1@I
ףp=
1@awC1���O?i[�q�&�?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_5/RealDiv_1(1
ףp=
1@9
ףp=
1@A
ףp=
1@I
ףp=
1@awC1���O?i��<�-�?�Unknown
��HostSelectV2"2model/gp_layer/truediv_2/softplus/forward/SelectV2(1
ףp=
1@9
ףp=
1@A
ףp=
1@I
ףp=
1@awC1���O?i�P���5�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_2_grad/Shape/TensorListPopBack(1D�l��	1@9D�l��	!@AD�l��	1@ID�l��	!@a�-�;�O?i^��x�=�?�Unknown
z�HostStridedSlice"model/gp_layer/strided_slice_13(1�A`���0@9�A`���0@A�A`���0@I�A`���0@a��5�O?i���qE�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_22_grad/TensorListPopBack(1��(\��0@9��(\�� @A��(\��0@I��(\�� @aw�O?i�)��1M�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_34(1�Zd{0@9�Zd{0@A�Zd{0@I�Zd{0@a@Bqk~N?i�k�=�T�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_20(1B`��"{0@9B`��"{ @AB`��"{0@IB`��"{ @aۃ1�}N?i��p\�?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_2/TensorListGetItem_grad/TensorListElementShape(1B`��"{0@9B`��"{ @AB`��"{0@IB`��"{ @aۃ1�}N?i#�6d�?�Unknown
o�HostSigmoid"Adam/gradients/Sigmoid_14(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@aۃ1�}N?iD/(��k�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_29(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@aۃ1�}N?iep�/Os�?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_20(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@aۃ1�}N?i��@��z�?�Unknown
��HostSelectV2"*model/gp_layer/softplus_1/forward/SelectV2(1B`��"{0@9B`��"{0@AB`��"{0@IB`��"{0@aۃ1�}N?i���(���?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1�ʡE�30@9�ʡE�30@A�ʡE�30@I�ʡE�30@a?�l��M?i�.(���?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_3_grad/Sum(1�ʡE�30@9�ʡE�3 @A�ʡE�30@I�ʡE�3 @a?�l��M?i�j����?�Unknown
s�HostConcatenateDataset"ConcatenateDataset(1�ʡE�30@9�ʡE�30@A�ʡE�30@I�ʡE�30@a?�l��M?i��ނ	��?�Unknown
��HostMatMul",gradient_tape/model/gp_layer/MatMul/MatMul_1(1�ʡE�30@9�ʡE�30@A�ʡE�30@I�ʡE�30@a?�l��M?i��9����?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_3(1�ʡE�30@9�ʡE�3 @A�ʡE�30@I�ʡE�3 @a?�l��M?i��i��?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_1(1j�t��/@9j�t��/@Aj�t��/@Ij�t��/@a��ۨ�uM?i�U��c��?�Unknown
��HostSelectV2"<gradient_tape/model/gp_layer/LogNormal_1/log_prob/SelectV2_1(1j�t��/@9j�t��/@Aj�t��/@Ij�t��/@a��ۨ�uM?in��=���?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_1/TensorListFromTensor(1j�t��/@9j�t��/@Aj�t��/@Ij�t��/@a��ۨ�uM?iO����?�Unknown
��HostMatrixBandPart"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/MatrixBandPart(1��Mb�/@9��Mb�@A��Mb�/@I��Mb�@a=��h/uM?i@���{��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_26/Mul(1��Mb�/@9��Mb�/@A��Mb�/@I��Mb�/@a=��h/uM?i1/�?���?�Unknown
��HostSelectV2",model/gp_layer/LogNormal_1/log_prob/SelectV2(1��Mb�/@9��Mb�/@A��Mb�/@I��Mb�/@a=��h/uM?i"e��6��?�Unknown
��HostMatMul"6gradient_tape/model/gp_layer/Tensordot/MatMul/MatMul_1(1}?5^�I/@9}?5^�I/@A}?5^�I/@I}?5^�I/@a���L?i㖛�r��?�Unknown
��HostMatMul"8gradient_tape/model/gp_layer/Tensordot_1/MatMul/MatMul_1(1}?5^�I/@9}?5^�I/@A}?5^�I/@I}?5^�I/@a���L?i�ȔM���?�Unknown
}�HostEqual")model/gp_layer/LogNormal_3/log_prob/Equal(1}?5^�I/@9}?5^�I/@A}?5^�I/@I}?5^�I/@a���L?ie������?�Unknown
f�HostAddN"Adam/gradients/AddN(1��K7I/@9��K7I/@A��K7I/@I��K7I/@a�Cä
�L?i6+7�'��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_18(1��K7I/@9��K7I/@A��K7I/@I��K7I/@a�Cä
�L?i\�3d��?�Unknown
��HostGreaterEqual")gradient_tape/model/gp_layer/GreaterEqual(1��K7I/@9��K7I/@A��K7I/@I��K7I/@a�Cä
�L?i،�v���?�Unknown
v�HostTile"#gradient_tape/model/gp_layer/Tile_9(1��K7I/@9��K7I/@A��K7I/@I��K7I/@a�Cä
�L?i��2���?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_12(1�|?5^�.@9�|?5^�@A�|?5^�.@I�|?5^�@aî��lL?iZ����?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_21(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@aî��lL?i#,�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_5(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@aî��lL?i�@�e.�?�Unknown
��HostSelectV2",model/gp_layer/LogNormal_3/log_prob/SelectV2(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@aî��lL?iml�I#�?�Unknown
j�HostMatMul"model/gp_layer/MatMul(1�|?5^�.@9�|?5^�.@A�|?5^�.@I�|?5^�.@aî��lL?i���d*�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_1_grad/BroadcastGradientArgs/TensorListPopBack(1�Q��+.@9�Q��+@A�Q��+.@I�Q��+@ajB���K?i���_1�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_24(1�Q��+.@9�Q��+.@A�Q��+.@I�Q��+.@ajB���K?i@�9Y8�?�Unknown
��HostAddV2"Amodel/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Internal_0_add_6(1�Q��+.@9�Q��+.@A�Q��+.@I�Q��+.@ajB���K?i�aiS?�?�Unknown
��HostSelectV2"2model/gp_layer/truediv_1/softplus/forward/SelectV2(1�Q��+.@9�Q��+.@A�Q��+.@I�Q��+.@ajB���K?ib2��MF�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_7(1^�I+.@9^�I+@A^�I+.@I^�I+@a���G�K?iX��GM�?�Unknown
��HostMul"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Square_grad/Mul_1(1^�I+.@9^�I+@A^�I+.@I^�I+@a���G�K?i�}��AT�?�Unknown
n�HostSigmoid"Adam/gradients/Sigmoid_1(1^�I+.@9^�I+.@A^�I+.@I^�I+.@a���G�K?iE���;[�?�Unknown
��HostSelectV2",model/gp_layer/LogNormal_2/log_prob/SelectV2(1^�I+.@9^�I+.@A^�I+.@I^�I+.@a���G�K?i�Ȅ�5b�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_8(1�&1��-@9�&1��@A�&1��-@I�&1��@a���X�dK?iV�i�?�Unknown
n�HostMaximum"model/gp_layer/Maximum_3(1�&1��-@9�&1��-@A�&1��-@I�&1��-@a���X�dK?i��/�o�?�Unknown
f�HostSum"model/gp_layer/Sum_2(1�&1��-@9�&1��-@A�&1��-@I�&1��-@a���X�dK?i6-�V�v�?�Unknown
f�HostExp"model/gp_layer/Exp_1(1\���(�-@9\���(�-@A\���(�-@I\���(�-@ah�#dK?i�M�_�}�?�Unknown
��HostSum"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/Sum_1(1�����-@9�����@A�����-@I�����@a1Aq�w�J?ijr}R��?�Unknown
f�HostExp"Adam/gradients/Exp_2(1Zd;�O-@9Zd;�O-@AZd;�O-@IZd;�O-@âmT��J?ih�}
��?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�����-@9�����-@A�����-@I�����-@ag�i��J?iٟL^�?�Unknown
��Host	ReverseV2"Omodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/ReverseV2(1X9��v~,@9X9��v~,@AX9��v~,@IX9��v~,@a0Y��[J?i��TY��?�Unknown
i�HostPack"model/gp_layer/stack_1(1X9��v~,@9X9��v~,@AX9��v~,@IX9��v~,@a0Y��[J?i[�K��?�Unknown
m�HostIteratorGetNext"IteratorGetNext(1ˡE��},@9ˡE��},@AˡE��},@IˡE��},@a�CUP`[J?i��(#���?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListReserve(1�v���+@9�v���@A�v���+@I�v���@a/�@�;�I?i������?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_3/TensorListGetItem_grad/TensorListElementShape(1�v���+@9�v���@A�v���+@I�v���@a/�@�;�I?i��r��?�Unknown
��HostLess"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/cond/_311/gradients/model/gp_layer/map/while_grad/Less(1�v���+@9��)g�@A�v���+@I��)g�@a/�@�;�I?i?ҏ��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_16(1�v���+@9�v���+@A�v���+@I�v���+@a/�@�;�I?ip"�^^��?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_2/TensorListFromTensor(1�v���+@9�v���+@A�v���+@I�v���+@a/�@�;�I?i�2�-���?�Unknown
��HostAssert"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/Assert/Assert(1�v���+@9�v���@A�v���+@I�v���@a/�@�;�I?i�B{�I��?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_2(1�v���+@9�v���@A�v���+@I�v���@a/�@�;�I?iS^˿��?�Unknown
��HostSelectV2"2model/gp_layer/truediv_4/softplus/forward/SelectV2(1�v���+@9�v���+@A�v���+@I�v���+@a/�@�;�I?i4cA�5��?�Unknown
��HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a�B,�SI?iEn�_���?�Unknown
{�HostMul")gradient_tape/model/gp_layer/mul_12/Mul_1(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a�B,�SI?iVy�%���?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_24/Sum(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a�B,�SI?ig�W�3��?�Unknown
l�HostMaximum"model/gp_layer/Maximum(1�K7�A`+@9�K7�A`+@A�K7�A`+@I�K7�A`+@a�B,�SI?ix�	����?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(19��v�_+@99��v�_+@A9��v�_+@I9��v�_+@a-�(��RI?i��kX���?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_4/RealDiv_1(1� �rh�*@9� �rh�*@A� �rh�*@I� �rh�*@a����H?i�����?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_13(17�A`��*@97�A`��*@A7�A`��*@I7�A`��*@a��x�H?i���D�?�Unknown
��HostReadVariableOp"6model/gp_layer/Reshape/identity/forward/ReadVariableOp(17�A`��*@97�A`��*@A7�A`��*@I7�A`��*@a��x�H?i��NQx�?�Unknown
��HostStridedSlice"|model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/strided_slice_1(17�A`��*@97�A`��@A7�A`��*@I7�A`��@a��x�H?i����?�Unknown
��HostMatrixBandPart"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatrixBandPart_grad/MatrixBandPart(1��(\�B*@9��(\�B@A��(\�B*@I��(\�B@a\A@�JH?i\�Ϣ��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_6_grad/TensorListPopBack(1��(\�B*@9��(\�B@A��(\�B*@I��(\�B@a\A@�JH?i,�V��?�Unknown
��HostGreaterEqual"+gradient_tape/model/gp_layer/GreaterEqual_3(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a����SJH?i���#�?�Unknown
u�HostMul"#gradient_tape/model/gp_layer/Mul_22(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a����SJH?i���)�?�Unknown
��HostTensorListStack"Egradient_tape/model/gp_layer/map/TensorArrayUnstack_4/TensorListStack(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a����SJH?iϯ	0�?�Unknown
x�HostGatherV2"!model/gp_layer/Tensordot/GatherV2(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a����SJH?i���6�?�Unknown
��HostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/strided_slice_2(15^�IB*@95^�IB@A5^�IB*@I5^�IB@a����SJH?i��?.<�?�Unknown
{�HostLog1p"'model/gp_layer/softplus_7/forward/Log1p(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a����SJH?ir��@B�?�Unknown
��HostAdd"Dmodel/likelihood_layer/model_likelihood_layer_Laplace/log_prob/sub_2(15^�IB*@95^�IB*@A5^�IB*@I5^�IB*@a����SJH?iS�iSH�?�Unknown
��HostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes/strided_slice_1(1��K7�A*@9��K7�A@A��K7�A*@I��K7�A@a�����IH?iD���eN�?�Unknown
��HostLess"qmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/cond/_126/model/gp_layer/map/while/Less(1�ʡE��)@9�1��y"@A�ʡE��)@I�1��y"@a���{��G?i���WT�?�Unknown
��HostAddN"8Adam/gradients/ArithmeticOptimizer/AddOpsRewrite_AddN_27(133333�)@933333�)@A33333�)@I33333�)@aZ�;/�G?i���IZ�?�Unknown
h�HostLess"Adam/gradients/Less_2(133333�)@933333�)@A33333�)@I33333�)@aZ�;/�G?iv���:`�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_3_grad/TensorListPopBack(133333�)@933333�@A33333�)@I33333�@aZ�;/�G?i7�[-,f�?�Unknown
y�HostDataset"#Iterator::Model::ParallelMapV2::Zip(1�&1�[�@9�&1�[�@A33333�)@I33333�)@aZ�;/�G?i��*�l�?�Unknown
t�HostMul""gradient_tape/model/gp_layer/Mul_1(133333�)@933333�)@A33333�)@I33333�)@aZ�;/�G?i���Dr�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_3(133333�)@933333�@A33333�)@I33333�@aZ�;/�G?iz��� x�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_1_grad/Sum(1��� ��)@9��� ��@A��� ��)@I��� ��@a�C����G?iK�G>�}�?�Unknown
x�HostSum"&gradient_tape/model/gp_layer/add/Sum_1(1��� ��)@9��� ��)@A��� ��)@I��� ��)@a�C����G?i~ƫ��?�Unknown
��HostConcatV2"=model/gp_layer/fill_triangular/forward/fill_triangular/concat(11�Z$)@91�Z$)@A1�Z$)@I1�Z$)@a���w
BG?i�sd.���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_16_grad/TensorListPopBack(1�p=
�#)@9�p=
�#@A�p=
�#)@I�p=
�#@aY��7�AG?imh�����?�Unknown
��HostSelectV2"2model/gp_layer/Squeeze_1/softplus/forward/SelectV2(1�p=
�#)@9�p=
�#)@A�p=
�#)@I�p=
�#)@aY��7�AG?i] �T��?�Unknown
��Host	Transpose"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/adjoint/matrix_transpose/transpose_grad/transpose(1���S#)@9���S#@A���S#)@I���S#@a���AG?i�P�<%��?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_1/RealDiv_1(1/�$��(@9/�$��(@A/�$��(@I/�$��(@a"³�F?i_Ak�Ԡ�?�Unknown
��HostStridedSlice"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/strided_slice(1/�$��(@9/�$��@A/�$��(@I/�$��@a"³�F?i�1�/���?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN(1�E����(@9�E����@A�E����(@I�E����@a�B�sl�F?ip!��3��?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_7(1�E����(@9�E����(@A�E����(@I�E����(@a�B�sl�F?i���?�Unknown
|�HostNeg"*gradient_tape/model/gp_layer/truediv_4/Neg(1�E����(@9�E����(@A�E����(@I�E����(@a�B�sl�F?i� /A���?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_19(1�E����(@9�E����@A�E����(@I�E����@a�B�sl�F?i#�K�A��?�Unknown
p�Host	Transpose"model/gp_layer/transpose(1�E����(@9�E����(@A�E����(@I�E����(@a�B�sl�F?i��h����?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/MatMul_1_grad/MatMul_1/TensorListPopBack(1�G�z�(@9�G�z�@A�G�z�(@I�G�z�@aW��3�F?iU�54���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_7_grad/Shape/TensorListPopBack(1�G�z�(@9�G�z�@A�G�z�(@I�G�z�@aW��3�F?i��qO��?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_1(1-���(@9-���@A-���(@I-���@a�����9F?iV�>����?�Unknown
^�HostCast"Adam/Cast_2(1�/�$(@9�/�$(@A�/�$(@I�/�$(@a!©�G9F?iǒ*3l��?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_35(1�/�$(@9�/�$(@A�/�$(@I�/�$(@a!©�G9F?i8}����?�Unknown
d�HostLog"model/gp_layer/Log(1�/�$(@9�/�$(@A�/�$(@I�/�$(@a!©�G9F?i�g׈��?�Unknown
��HostMul"Qgradient_tape/model/gp_layer/LogNormal_1/log_prob/LogNormal_Normal_1/log_prob/Mul(1��ʡ(@9��ʡ(@A��ʡ(@I��ʡ(@a��o�8F?i*Q�
��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_11/Mul(1��ʡ(@9��ʡ(@A��ʡ(@I��ʡ(@a��o�8F?i�::>���?�Unknown
\�HostPow"
Adam/Pow_1(1���Kw'@9���Kw'@A���Kw'@I���Kw'@a�A��"�E?i�����?�Unknown
g�HostExp"Adam/gradients/Exp_10(1���Kw'@9���Kw'@A���Kw'@I���Kw'@a�A��"�E?iK����?�Unknown
i�HostAddN"Adam/gradients/AddN_11(1X9��v'@9X9��v'@AX9��v'@IX9��v'@a�����E?i������?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_15(1X9��v'@9X9��v@AX9��v'@IX9��v@a�����E?i΅$Z�?�Unknown
��HostPack"~model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_2/values_1(1X9��v'@9X9��v@AX9��v'@IX9��v@a�����E?in��N�
�?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_1/TensorListGetItem_grad/TensorListElementShape(1�� �r�&@9�� �r�@A�� �r�&@I�� �r�@a���'�0E?i��z��?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem_grad/TensorListSetItem(1-����&@9-����@A-����&@I-����@a�}�0E?i�q��_�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_30(1-����&@9-����&@A-����&@I-����&@a�}�0E?i Q�Ы�?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_9(1-����&@9-����&@A-����&@I-����&@a�}�0E?ia0(���?�Unknown
{�HostMul")gradient_tape/model/gp_layer/mul_20/Mul_1(1-����&@9-����&@A-����&@I-����&@a�}�0E?i�bD%�?�Unknown
m�HostSquare"model/gp_layer/Square_10(1-����&@9-����&@A-����&@I-����&@a�}�0E?i��4�*�?�Unknown
g�HostMul"model/gp_layer/mul_24(1-����&@9-����&@A-����&@I-����&@a�}�0E?i$��U�/�?�Unknown
g�HostMul"model/gp_layer/mul_25(1-����&@9-����&@A-����&@I-����&@a�}�0E?ie�w(5�?�Unknown
n�HostRealDiv"model/gp_layer/truediv_5(1-����&@9-����&@A-����&@I-����&@a�}�0E?i��I�t:�?�Unknown
i�HostLess"Adam/gradients/Less_15(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a�h#`�D?i�fR��?�?�Unknown
i�HostAddV2"model/gp_layer/add_16(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a�h#`�D?i�@[��D�?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_4(1+�Y&@9+�Y@A+�Y&@I+�Y@a�h#`�D?id��I�?�Unknown
�HostSelectV2"(model/gp_layer/softplus/forward/SelectV2(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a�h#`�D?i&�l� O�?�Unknown
�HostSoftplus"(model/gp_layer/softplus/forward/Softplus(1+�Y&@9+�Y&@A+�Y&@I+�Y&@a�h#`�D?iF�uLT�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_21(1j�t�X&@9j�t�X@Aj�t�X&@Ij�t�X@a��d��D?iw�.
wY�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_20/Mul(1j�t�X&@9j�t�X&@Aj�t�X&@Ij�t�X&@a��d��D?i����^�?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write_1/TensorListSetItem_grad/TensorListSetItem(1
ףp=�%@9
ףp=�@A
ףp=�%@I
ףp=�@aLT_;(D?i�V��c�?�Unknown
��HostNeg"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/Neg(1
ףp=�%@9
ףp=�@A
ףp=�%@I
ףp=�@aLT_;(D?i�+�!�h�?�Unknown
x�HostSum"&gradient_tape/model/gp_layer/add_5/Sum(1
ףp=�%@9
ףp=�%@A
ףp=�%@I
ףp=�%@aLT_;(D?i� o0�m�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_19/Mul(1
ףp=�%@9
ףp=�%@A
ףp=�%@I
ףp=�%@aLT_;(D?i��F?�r�?�Unknown
��HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1}?5^��%@9}?5^��%@A}?5^��%@I}?5^��%@a�BP�'D?i���/�w�?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_2/TensorListGetItem_grad/TensorListReserve(1}?5^��%@9}?5^��@A}?5^��%@I}?5^��@a�BP�'D?i�}V �|�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_21_grad/TensorListPopBack(1}?5^��%@9}?5^��@A}?5^��%@I}?5^��@a�BP�'D?i�Q���?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_13(1}?5^��%@9}?5^��%@A}?5^��%@I}?5^��%@a�BP�'D?i�%f��?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_22/Sum(1}?5^��%@9}?5^��%@A}?5^��%@I}?5^��%@a�BP�'D?i�������?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_8(1}?5^��%@9}?5^��@A}?5^��%@I}?5^��@a�BP�'D?i�u���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_4_grad/Shape/TensorListPopBack(1�Zd;%@9�Zd;@A�Zd;%@I�Zd;@a��?��C?i����?�Unknown
z�HostNeg"(gradient_tape/model/gp_layer/truediv/Neg(1�Zd;%@9�Zd;%@A�Zd;%@I�Zd;%@a��?��C?i�m��ך�?�Unknown
g�HostPack"model/gp_layer/stack(1�Zd;%@9�Zd;%@A�Zd;%@I�Zd;%@a��?��C?i�=j����?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_19(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@aK�;[��C?i��ک��?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem_grad/TensorListElementShape(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@aK�;[��C?i����?�Unknown
��HostMatrixBandPart"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/MatrixBandPart(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@aK�;[��C?i��n�{��?�Unknown
p�HostSelectV2"Adam/gradients/SelectV2_1(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?iryŐd��?�Unknown
��HostTensorListFromTensor"Hgradient_tape/model/gp_layer/map/TensorArrayV2Stack/TensorListFromTensor(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?icHxM��?�Unknown
y�HostSum"'gradient_tape/model/gp_layer/mul_23/Sum(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?iTs_6��?�Unknown
t�HostMul""gradient_tape/model/gp_layer/mul_7(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?iE��F��?�Unknown
n�HostMaximum"model/gp_layer/Maximum_2(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?i6� .��?�Unknown
l�HostSquare"model/gp_layer/Square_1(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?i'�w���?�Unknown
f�HostTile"model/gp_layer/Tile(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?iS�����?�Unknown
��HostEqual"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes/Equal(1{�G�:%@9{�G�:@A{�G�:%@I{�G�:@aK�;[��C?i	"%����?�Unknown
y�HostStridedSlice"model/gp_layer/strided_slice_9(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?i��{˫��?�Unknown
��HostSelectV2"2model/gp_layer/truediv_3/softplus/forward/SelectV2(1{�G�:%@9{�G�:%@A{�G�:%@I{�G�:%@aK�;[��C?i�Ҳ���?�Unknown
��Host	ZerosLike"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write_1/TensorListSetItem_grad/zeros_like(1y�&1�$@9y�&1�@Ay�&1�$@Iy�&1�@a�A'�xC?i����\��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_10_grad/Shape/TensorListPopBack(1y�&1�$@9y�&1�@Ay�&1�$@Iy�&1�@a�A'�xC?i�So$��?�Unknown
��HostStridedSlice"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/strided_slice_1(1y�&1�$@9y�&1�@Ay�&1�$@Iy�&1�@a�A'�xC?i[DM���?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_15/Mul(1y�&1�$@9y�&1�$@Ay�&1�$@Iy�&1�$@a�A'�xC?i+�i+���?�Unknown
��HostAddN">model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_add_14(1y�&1�$@9y�&1�$@Ay�&1�$@Iy�&1�$@a�A'�xC?i���	|��?�Unknown
h�HostAddN"Adam/gradients/AddN_1(1�Q���$@9�Q���$@A�Q���$@I�Q���$@aJ�#W�C?i�ye�C��?�Unknown
��HostTensorListElementShape"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem_grad/TensorListElementShape(1�Q���$@9�Q���@A�Q���$@I�Q���@aJ�#W�C?i�B;��?�Unknown
n�HostSigmoid"Adam/gradients/Sigmoid_8(1�Q���$@9�Q���$@A�Q���$@I�Q���$@aJ�#W�C?i�I��?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_3/RealDiv_1(1�Q���$@9�Q���$@A�Q���$@I�Q���$@aJ�#W�C?i���
�?�Unknown
f�HostExp"Adam/gradients/Exp_3(1w��/$@9w��/$@Aw��/$@Iw��/$@a��S�B?i/���A�?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_3(1w��/$@9w��/$@Aw��/$@Iw��/$@a��S�B?i�]в��?�Unknown
s�Host	ZerosLike"Adam/gradients/zeros_like_4(1w��/$@9w��/$@Aw��/$@Iw��/$@a��S�B?i�"Ň��?�Unknown
g�HostMul"model/gp_layer/mul_22(1w��/$@9w��/$@Aw��/$@Iw��/$@a��S�B?i?�\6�?�Unknown
��HostAddV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/add(1�&1�$@9�&1�@A�&1�$@I�&1�@a��ښB?i �^�!�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/triangular_solve/MatrixTriangularSolve_grad/Sum(1�&1�$@9�&1�@A�&1�$@I�&1�@a��ښB?i�nʃ&�?�Unknown
��HostTensorListFromTensor"Jgradient_tape/model/gp_layer/map/TensorArrayV2Stack_1/TensorListFromTensor(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a��ښB?i�2��*+�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_24/Mul(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a��ښB?iC�L7�/�?�Unknown
��HostMatrixBandPart"Tmodel/gp_layer/MatrixBandPart/fill_triangular/forward/fill_triangular/MatrixBandPart(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a��ښB?i���w4�?�Unknown
f�HostSum"model/gp_layer/Sum_5(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a��ښB?i�}��9�?�Unknown
g�HostMul"model/gp_layer/mul_26(1�&1�$@9�&1�$@A�&1�$@I�&1�$@a��ښB?i�A;[�=�?�Unknown
��HostMatrixDiagPartV3"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/diag_part(1\���($@9\���(@A\���($@I\���(@aHDSa�B?iW��kB�?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_1/TensorListGetItem_grad/TensorListReserve(1u�V�#@9u�V�@Au�V�#@Iu�V�@aw@�/B?i��S��F�?�Unknown
h�HostAddN"Adam/gradients/AddN_3(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?i���lwK�?�Unknown
i�HostLess"Adam/gradients/Less_12(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?i)A;�O�?�Unknown
��HostAddN"/Adam/gradients/PartitionedCall/gradients/AddN_3(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?i���ǂT�?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_11(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@a��εB?ik�"uY�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_11(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?i}�"�]�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_17(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?i�;
�b�?�Unknown
��HostDynamicStitch"*gradient_tape/model/gp_layer/DynamicStitch(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?iN�}}�f�?�Unknown
��HostTensorListFromTensor"<model/gp_layer/map/TensorArrayUnstack_3/TensorListFromTensor(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?i��*k�?�Unknown
��HostBroadcastTo"zmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/BroadcastTo_1(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@a��εB?i�weؤo�?�Unknown
��HostTensorListPushBack"model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@a��εB?i16م*t�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_21(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@a��εB?i��L3�x�?�Unknown
��HostPack"~model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_5/values_1(1����ҍ#@9����ҍ@A����ҍ#@I����ҍ@a��εB?is���5}�?�Unknown
g�HostMul"model/gp_layer/mul_12(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?ir4����?�Unknown
f�HostMul"model/gp_layer/mul_2(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?i�0�;A��?�Unknown
�HostEqual"+model/likelihood_layer/assert_equal_1/Equal(1����ҍ#@9����ҍ#@A����ҍ#@I����ҍ#@a��εB?iV��Ɗ�?�Unknown
��HostSum"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/add_grad/Sum_1(1Zd;�O�#@9Zd;�O�@AZd;�O�#@IZd;�O�@a����<B?i�?xL��?�Unknown
u�HostReadVariableOp"Adam/Cast/ReadVariableOp(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?i�f����?�Unknown
i�HostLess"Adam/gradients/Less_13(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?i ����?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_3(1��"���"@9��"���@A��"���"@I��"���@au�
��A?i��ez��?�Unknown
��HostGreaterEqual"+gradient_tape/model/gp_layer/GreaterEqual_1(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?i�J	ߠ�?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_6(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?i�L��C��?�Unknown
��HostTensorListStack"5model/gp_layer/map/TensorArrayV2Stack/TensorListStack(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?i�Q���?�Unknown
g�HostMul"model/gp_layer/mul_21(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?i�����?�Unknown
y�HostStridedSlice"model/gp_layer/strided_slice_1(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?iyU�q��?�Unknown
��HostSoftplus"2model/gp_layer/truediv_3/softplus/forward/Softplus(1��"���"@9��"���"@A��"���"@I��"���"@au�
��A?i�2�>ֶ�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/BroadcastTo_4_grad/BroadcastGradientArgs/TensorListPopBack(1X9��v�"@9X9��v�@AX9��v�"@IX9��v�@aC���A?i��:��?�Unknown
��HostTensorListReserve"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_4/TensorListGetItem_grad/TensorListReserve(1X9��v�"@9X9��v�@AX9��v�"@IX9��v�@aC���A?i��}J���?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_4(1X9��v�"@9X9��v�"@AX9��v�"@IX9��v�"@aC���A?i:\p���?�Unknown
��HostConcatV2"umodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/concat_5(1X9��v�"@9X9��v�@AX9��v�"@IX9��v�@aC���A?i�cVh��?�Unknown
l�HostRealDiv"model/gp_layer/truediv(1X9��v�"@9X9��v�"@AX9��v�"@IX9��v�"@aC���A?i\�U����?�Unknown
��HostTensorListGetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Read/TensorListGetItem(1q=
ףp"@9q=
ףp@Aq=
ףp"@Iq=
ףp@a@?Ն�A?i������?�Unknown
��HostAddV2"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/Cholesky_grad/add(1㥛� p"@9㥛� p@A㥛� p"@I㥛� p@aـ�FlA?i7�0T��?�Unknown
d�HostMul"model/gp_layer/mul(1㥛� p"@9㥛� p"@A㥛� p"@I㥛� p"@aـ�FlA?il��˗��?�Unknown
��HostSelectV2"*model/gp_layer/softplus_5/forward/SelectV2(1㥛� p"@9㥛� p"@A㥛� p"@I㥛� p"@aـ�FlA?i̟�f���?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_11_grad/Shape/TensorListPopBack(1V-��o"@9V-��o@AV-��o"@IV-��o@at���A?i=S����?�Unknown
t�HostSum""gradient_tape/model/gp_layer/Sum_1(1V-��o"@9V-��o"@AV-��o"@IV-��o"@at���A?i�p`b��?�Unknown
��Host	Transpose"0gradient_tape/model/gp_layer/transpose/transpose(1V-��o"@9V-��o"@AV-��o"@IV-��o"@at���A?i�1ݥ��?�Unknown
��HostLess"smodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/cond/_126/model/gp_layer/map/while/Less_1(1�z�G�!@9ףp=
�@A�z�G�!@Iףp=
�@a> ��G�@?i_io���?�Unknown
i�HostAddN"Adam/gradients/AddN_25(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i������?�Unknown
f�HostMul"Adam/gradients/mul_4(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i��3V��?�Unknown
q�Host	ZerosLike"Adam/gradients/zeros_like(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?iOt��/��?�Unknown
i�HostRandomShuffle"RandomShuffle(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i�"U=R��?�Unknown
��Host
Reciprocal"Rgradient_tape/model/gp_layer/LogNormal_1/log_prob/LogNormal_exp/inverse/Reciprocal(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i���t�?�Unknown
��HostSlice"Jgradient_tape/model/gp_layer/fill_triangular/forward/fill_triangular/Slice(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i?v$��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_16/Mul(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i�-���?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_23/Mul(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i�ۗ��?�Unknown
��HostAddN">model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_Leaf_1_add_18(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i/�(��?�Unknown
��HostAddV2"6model/gp_layer/ArithmeticOptimizer/AddOpsRewrite_add_1(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i8�� �?�Unknown
h�HostSqrt"model/gp_layer/Sqrt_1(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i��IfC�?�Unknown
f�HostSum"model/gp_layer/Sum_4(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i���e �?�Unknown
��HostTensorListSetItem"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorArrayV2Write/TensorListSetItem(1T㥛��!@9T㥛��@AT㥛��!@IT㥛��@a�A�BΉ@?ioCkM�$�?�Unknown
��HostReadVariableOp".model/gp_layer/softplus/forward/ReadVariableOp(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i�����(�?�Unknown
��HostReadVariableOp"0model/gp_layer/softplus_6/forward/ReadVariableOp(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i��4�,�?�Unknown
��HostSoftplus"2model/gp_layer/truediv_4/softplus/forward/Softplus(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i_N��0�?�Unknown
��HostStridedSlice"&model/likelihood_layer/strided_slice_2(1T㥛��!@9T㥛��!@AT㥛��!@IT㥛��!@a�A�BΉ@?i���5�?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_13_grad/Shape/TensorListPopBack(1�K7�A�!@9�K7�A�@A�K7�A�!@I�K7�A�@as��U�@?i��p49�?�Unknown
i�HostAddN"Adam/gradients/AddN_24(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a=��~�@?i@SN�5=�?�Unknown
��HostTensorListSetItem"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Write/TensorListSetItem_grad/TensorListSetItem(1R���Q!@9R���Q@AR���Q!@IR���Q@a=��~�@?ip��E7A�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_33(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a=��~�@?i���8E�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_22/Mul(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a=��~�@?i�Nm:I�?�Unknown
x�HostMul"&gradient_tape/model/gp_layer/mul_5/Mul(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a=��~�@?i �̄;M�?�Unknown
~�HostSum",gradient_tape/model/gp_layer/truediv_2/Sum_1(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a=��~�@?i0�,�<Q�?�Unknown
��HostStridedSlice"Dmodel/gp_layer/fill_triangular/forward/fill_triangular/strided_slice(1R���Q!@9R���Q!@AR���Q!@IR���Q!@a=��~�@?i`J�Y>U�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_22(1R���Q!@9R���Q@AR���Q!@IR���Q@a=��~�@?i����?Y�?�Unknown
i�HostAddN"Adam/gradients/AddN_10(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a��>0@?iћ�A]�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_19(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a��>0@?iD\Ba�?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_18/Mul(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a��>0@?iS��Ce�?�Unknown
��HostRealDiv"0gradient_tape/model/gp_layer/truediv_2/RealDiv_2(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a��>0@?i��*�Di�?�Unknown
��HostEqual"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/assert_shapes_1/Equal(1� �rhQ!@9� �rhQ@A� �rhQ!@I� �rhQ@a��>0@?i�<:@Fm�?�Unknown
g�HostMul"model/gp_layer/mul_19(1� �rhQ!@9� �rhQ!@A� �rhQ!@I� �rhQ!@a��>0@?i�I�Gq�?�Unknown
t�Host	ZerosLike"Adam/gradients/zeros_like_22(1P��n� @9P��n� @AP��n� @IP��n� @aA� u	??i&�x�'u�?�Unknown
_�HostGatherV2"GatherV2(1P��n� @9P��n� @AP��n� @IP��n� @aA� u	??i6-�Ny�?�Unknown
��HostAddV2"Cmodel/gp_layer/LogNormal_2/log_prob/LogNormal_Normal_1/log_prob/add(1P��n� @9P��n� @AP��n� @IP��n� @aA� u	??iF�կ�|�?�Unknown
g�HostSum"model/gp_layer/Sum_11(1P��n� @9P��n� @AP��n� @IP��n� @aA� u	??iVuɀ�?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_11(1P��n� @9P��n�@AP��n� @IP��n�@aA� u	??if3r���?�Unknown
��HostAddN"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/AddN_1(1��(\�� @9��(\��@A��(\�� @I��(\��@aw�??i������?�Unknown
��HostTensorListLength"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorArrayV2Read_1/TensorListGetItem_grad/TensorListLength(1��(\�� @9��(\��@A��(\�� @I��(\��@aw�??i�_��i��?�Unknown
g�HostMul"Adam/gradients/mul_13(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??i��:J��?�Unknown
��HostDynamicStitch",gradient_tape/model/gp_layer/DynamicStitch_1(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??iꥭ}*��?�Unknown
{�HostSum")gradient_tape/model/gp_layer/add_16/Sum_1(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??iI��
��?�Unknown
y�HostMul"'gradient_tape/model/gp_layer/mul_21/Mul(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??i,�j��?�Unknown
t�HostMul""gradient_tape/model/gp_layer/mul_9(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??iM�IF˟�?�Unknown
��HostRealDiv"Gmodel/gp_layer/LogNormal_2/log_prob/LogNormal_Normal_1/log_prob/truediv(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??in2(����?�Unknown
��HostTensorListPushBack"�model/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/TensorListPushBack_6(1��(\�� @9��(\��@A��(\�� @I��(\��@aw�??i��̋��?�Unknown
��HostSub"rmodel/gp_layer/map/while/StatefulPartitionedCall/model/gp_layer/map/while/body/_127/model/gp_layer/map/while/sub_3(1��(\�� @9��(\��@A��(\�� @I��(\��@aw�??i�x�l��?�Unknown
w�HostStridedSlice"model/gp_layer/strided_slice(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??i��QL��?�Unknown
n�HostRealDiv"model/gp_layer/truediv_2(1��(\�� @9��(\�� @A��(\�� @I��(\�� @aw�??i򾢔,��?�Unknown
��HostTensorListPopBack"�Adam/gradients/PartitionedCall/gradients/model/gp_layer/map/while_grad/model/gp_layer/map/while_grad/body/_312/gradients/model/gp_layer/map/while_grad/gradients/model/gp_layer/map/while/TensorListPushBack_12_grad/TensorListPopBack(1NbX94 @9NbX94@ANbX94 @INbX94@a
����=?i�]����?�Unknown
^�HostCast"Adam/Cast_3(1�ʡE�3 @9�ʡE�3 @A�ʡE�3 @I�ʡE�3 @a?�l��=?i��M&���?�Unknown
f�HostExp"Adam/gradients/Exp_9(1�ʡE�3 @9�ʡE�3 @A�ʡE�3 @I�ʡE�3 @a?�l��=?i��_j��?�Unknown2CPU