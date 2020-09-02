# Copyright 2020 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kubric.assets import utils

# a large list of cryptomatte ids that were manually extracted
# (100 may be overkill. But I did have the first 29 succeed only to fail at Object_30)
name_to_crypto = [
    ("Static", 3498399415), ("Object_00", 991243257), ("Object_01", 2711523813),
    ("Object_02", 2458670877), ("Object_03", 3322849070), ("Object_04", 1185212285),
    ("Object_05", 2611205165), ("Object_06", 2826885510), ("Object_07", 619118036),
    ("Object_08", 3366639678), ("Object_09", 2866447012), ("Object_10", 3303165227),
    ("Object_11", 2274438809), ("Object_12", 4248664771), ("Object_13", 922688289),
    ("Object_14", 1062878951), ("Object_15", 3996663913), ("Object_16", 1747592571),
    ("Object_17", 2656021041), ("Object_18", 4237058216), ("Object_19", 3976614580),
    ("Object_20", 2985862269), ("Object_21", 3285582496), ("Object_22", 1752553109),
    ("Object_23", 3147606985), ("Object_24", 4272928739), ("Object_25", 3964064050),
    ("Object_26", 1190004375), ("Object_27", 3384364420), ("Object_28", 2435969472),
    ("Object_29", 2492285371), ("Object_30", 2135499230), ("Object_31", 2122842703),
    ("Object_32", 825185619), ("Object_33", 1321257118), ("Object_34", 1307903920),
    ("Object_35", 4181187753), ("Object_36", 1703180566), ("Object_37", 3578279029),
    ("Object_38", 204708733), ("Object_39", 576564171), ("Object_40", 211896534),
    ("Object_41", 12644183), ("Object_42", 3384229550), ("Object_43", 3697986911),
    ("Object_44", 3785502810), ("Object_45", 4027686760), ("Object_46", 3559797246),
    ("Object_47", 4200427088), ("Object_48", 2786625024), ("Object_49", 1700704528),
    ("Object_50", 566993987), ("Object_51", 296172981), ("Object_52", 2763560336),
    ("Object_53", 2055044291), ("Object_54", 1839783943), ("Object_55", 1605948219),
    ("Object_56", 1443214841), ("Object_57", 3392711039), ("Object_58", 661628604),
    ("Object_59", 178932917), ("Object_60", 3816579188), ("Object_61", 2135458370),
    ("Object_62", 538166773), ("Object_63", 1449002966), ("Object_64", 3072669936),
    ("Object_65", 1263490313), ("Object_66", 4218338862), ("Object_67", 1436130700),
    ("Object_68", 3138848958), ("Object_69", 2027132096), ("Object_70", 1609465218),
    ("Object_71", 1481631480), ("Object_72", 359154495), ("Object_73", 3146125111),
    ("Object_74", 4042886601), ("Object_75", 3958756063), ("Object_76", 683500781),
    ("Object_77", 866742119), ("Object_78", 3423258758), ("Object_79", 2200031569),
    ("Object_80", 2467585833), ("Object_81", 1372927894), ("Object_82", 366391357),
    ("Object_83", 2157990704), ("Object_84", 763970710), ("Object_85", 3081330126),
    ("Object_86", 1790370758), ("Object_87", 1862659878), ("Object_88", 4121469279),
    ("Object_89", 22464357), ("Object_90", 1612373827), ("Object_91", 3504678270),
    ("Object_92", 2209378211), ("Object_93", 160191028), ("Object_94", 3882592811),
    ("Object_95", 1938643046), ("Object_96", 4195052494), ("Object_97", 2784341116),
    ("Object_98", 2907465629), ("Object_99", 25278216)]


def test_mm3hash():
    for name, expected in name_to_crypto:
        assert utils.mm3hash(name) == expected
