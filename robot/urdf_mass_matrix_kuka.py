import torch
import torch.jit as jit


def compute_mass_matrix_urdf(q):

    horizon = q.shape[1]
    Mq = torch.empty(horizon, 7, 7, dtype=torch.float32).to("cuda")
    q1 = q[0, :]
    q2 = q[1, :]
    q3 = q[2, :]
    q4 = q[3, :]
    q5 = q[4, :]
    q6 = q[5, :]
    q7 = q[6, :]

    c1, s1 = torch.cos(q1), torch.sin(q1)
    c2, s2 = torch.cos(q2), torch.sin(q2)
    c3, s3 = torch.cos(q3), torch.sin(q3)
    c4, s4 = torch.cos(q4), torch.sin(q4)
    c5, s5 = torch.cos(q5), torch.sin(q5)
    c6, s6 = torch.cos(q6), torch.sin(q6)
    # c7, s7 = torch.cos(q7), torch.sin(q7)

    var1 = s1
    var2 = 1.70704e-07
    var3 = c1
    var4 = -0.0327465
    var5 = 2.30344
    var6 = var5 * (var4 * var1)
    var7 = 0.0736557
    var8 = 1.26775e-06
    var9 = var5 * ((var7 * var1) - (var8 * var3))
    var10 = c2
    var11 = 0.0156098
    var12 = var10 * var1
    var13 = 4.7548e-08
    var14 = s2
    var15 = var14 * var1
    var16 = 1.17852e-07
    var17 = -0.0233298
    var18 = 2.30342
    var19 = -1.40921e-06
    var20 = var18 * ((var19 * var15) + (var17 * var12))
    var21 = 0.11815
    var22 = 0.1915
    var23 = var22 * var1
    var24 = var10 * var23
    var25 = var18 * (var24 + ((var21 * var12) - (var19 * var3)))
    var26 = c3
    var27 = 0.0142337
    var28 = s3
    var29 = (var26 * var12) - (var28 * var3)
    var30 = -5.89296e-08
    var31 = -1.56827e-07
    var32 = (var28 * var12) + (var26 * var3)
    var33 = 0.0327442
    var34 = 2.30344
    var35 = 1.12239e-07
    var36 = var14 * var23
    var37 = 0.2085
    var38 = var36 + (var37 * var15)
    var39 = var28 * var38
    var40 = var34 * (((var35 * var15) + (var33 * var29)) - var39)
    var41 = 0.0736588
    var42 = var24 + (var37 * var12)
    var43 = var34 * (var42 + ((var41 * var29) - (var35 * var32)))
    var44 = c4
    var45 = 0.00880807
    var46 = s4
    var47 = (var44 * var29) - (var46 * var15)
    var48 = 1.2282e-07
    var49 = (var44 * var15) + (var46 * var29)
    var50 = -5.66844e-08
    var51 = 0.0207752
    var52 = 1.6006
    var53 = -6.00825e-07
    var54 = var52 * (((var53 * var49) + (var51 * var47)) - var39)
    var55 = 0.0862054
    var56 = var26 * var38
    var57 = var56 + (var22 * var15)
    var58 = var42 + (var22 * var29)
    var59 = (var46 * var57) - (var44 * var58)
    var60 = var52 * (var59 - ((var55 * var47) - (var53 * var32)))
    var61 = c5
    var62 = 0.0298541
    var63 = s5
    var64 = (var61 * var47) + (var63 * var32)
    var65 = -3.97659e-09
    var66 = -1.71667e-09
    var67 = (var61 * var32) - (var63 * var47)
    var68 = -0.00451754
    var69 = 1.49302
    var70 = (var44 * var57) + (var46 * var58)
    var71 = 0.1985
    var72 = var70 + (var71 * var49)
    var73 = (var63 * var72) - (var61 * var39)
    var74 = -2.64519e-08
    var75 = var69 * (var73 + ((var74 * var49) + (var68 * var64)))
    var76 = -0.00295325
    var77 = var59 - (var71 * var47)
    var78 = var69 * (var77 - ((var76 * var64) - (var74 * var67)))
    var79 = c6
    var80 = 0.0417909
    var81 = s6
    var82 = (var79 * var64) - (var81 * var49)
    var83 = 1.11022e-16
    var84 = 0.108688
    var85 = 2.77556e-17
    var86 = (var79 * var49) + (var81 * var64)
    var87 = var84 * (var73 + ((var85 * var86) + (var83 * var82)))
    var88 = -0.0158147
    var89 = 0.078
    var90 = var77 - (var89 * var64)
    var91 = (var61 * var72) + (var63 * var39)
    var92 = var91 + (var89 * var49)
    var93 = var84 * (((var79 * var90) + (var81 * var92)) - ((var88 * var82) - (var85 * var67)))
    var94 = (var80 * var82) + ((var83 * var87) - (var88 * var93))
    var95 = var84 * (((var81 * var90) - (var79 * var92)) - ((var83 * var67) + (var88 * var86)))
    var96 = ((var88 * var95) - (var85 * var87)) - (var80 * var86)
    var97 = (var81 * var95) + (var79 * var93)
    var98 = ((((var62 * var64) - (var65 * var49)) + (var66 * var67)) + ((var68 * var75) - (var76 * var78))) + (
        ((var79 * var94) - (var81 * var96)) - (var89 * var97)
    )
    var99 = -2.53647e-05
    var100 = 0.0323627
    var101 = var69 * (var91 + ((var68 * var67) + (var76 * var49)))
    var102 = 0.0700757
    var103 = (var102 * var67) + ((var85 * var93) - (var83 * var95))
    var104 = ((((var66 * var64) - (var99 * var49)) + (var100 * var67)) + ((var74 * var78) + (var68 * var101))) + var103
    var105 = var78 + var97
    var106 = ((((var45 * var47) - (var48 * var49)) + (var50 * var32)) + ((var51 * var54) - (var55 * var60))) + (
        ((var61 * var98) - (var63 * var104)) - (var71 * var105)
    )
    var107 = 0.0081352
    var108 = 0.00261444
    var109 = var52 * (var70 + ((var51 * var32) + (var55 * var49)))
    var110 = 0.0299835
    var111 = (var79 * var95) - (var81 * var93)
    var112 = ((((var65 * var64) - (var110 * var49)) + (var99 * var67)) - ((var76 * var101) + (var74 * var75))) + (
        ((var81 * var94) + (var79 * var96)) + (var89 * var111)
    )
    var113 = var111 - var101
    var114 = var75 + var87
    var115 = (var61 * var113) - (var63 * var114)
    var116 = ((((var48 * var47) - (var107 * var49)) + (var108 * var32)) - ((var55 * var109) + (var53 * var54))) + (
        var112 + (var71 * var115)
    )
    var117 = var115 - var109
    var118 = var60 + var105
    var119 = (var46 * var117) + (var44 * var118)
    var120 = ((((var27 * var29) - (var30 * var15)) + (var31 * var32)) + ((var33 * var40) + (var41 * var43))) + (
        ((var44 * var106) - (var46 * var116)) - (var22 * var119)
    )
    var121 = -0.00228056
    var122 = 0.00424817
    var123 = var34 * (var56 + ((var33 * var32) + (var41 * var15)))
    var124 = 0.00359712
    var125 = ((((var50 * var47) - (var108 * var49)) + (var124 * var32)) + ((var53 * var60) + (var51 * var109))) + (
        (var63 * var98) + (var61 * var104)
    )
    var126 = ((((var31 * var29) - (var121 * var15)) + (var122 * var32)) + ((var33 * var123) - (var35 * var43))) + var125
    var127 = var119 - var43
    var128 = ((((var11 * var12) - (var13 * var15)) + (var16 * var3)) + ((var17 * var20) + (var21 * var25))) + (
        ((var26 * var120) + (var28 * var126)) - (var37 * var127)
    )
    var129 = 0.0153477
    var130 = -0.00319216
    var131 = var18 * (var36 + ((var17 * var3) + (var21 * var15)))
    var132 = 0.0141316
    var133 = (var44 * var117) - (var46 * var118)
    var134 = ((((var30 * var29) - (var132 * var15)) + (var121 * var32)) - ((var41 * var123) + (var35 * var40))) + (
        ((var46 * var106) + (var44 * var116)) + (var22 * var133)
    )
    var135 = (var26 * (var133 - var123)) + (var28 * (var40 + (var54 + ((var63 * var113) + (var61 * var114)))))
    var136 = ((((var13 * var12) - (var129 * var15)) + (var130 * var3)) - ((var21 * var131) + (var19 * var20))) + (
        var134 + (var37 * var135)
    )
    var137 = var135 - var131
    var138 = var127 - var25
    var139 = var5 * (var4 * var3)
    var140 = ((((var16 * var12) - (var130 * var15)) + (0.00440714 * var3)) + ((var17 * var131) - (var19 * var25))) + (
        (var26 * var126) - (var28 * var120)
    )
    var141 = var18 * ((var19 * var10) - (var17 * var14))
    var142 = -0.1915
    var143 = var142 * var14
    var144 = var18 * ((var21 * var14) - var143)
    var145 = var142 * var10
    var146 = var145 - (var37 * var10)
    var147 = var28 * var146
    var148 = var26 * var14
    var149 = var34 * (var147 - ((var33 * var148) - (var35 * var10)))
    var150 = (var37 * var14) - var143
    var151 = var28 * var14
    var152 = var34 * (var150 - ((var35 * var151) - (var41 * var148)))
    var153 = (var46 * var148) - (var44 * var10)
    var154 = (var44 * var148) + (var46 * var10)
    var155 = var52 * (var147 - ((var53 * var153) + (var51 * var154)))
    var156 = var150 + (var22 * var148)
    var157 = var26 * var146
    var158 = var157 - (var22 * var10)
    var159 = (var44 * var156) - (var46 * var158)
    var160 = var52 * (var159 - ((var53 * var151) - (var55 * var154)))
    var161 = (var61 * var154) + (var63 * var151)
    var162 = (var63 * var154) - (var61 * var151)
    var163 = (var44 * var158) + (var46 * var156)
    var164 = var163 + (var71 * var153)
    var165 = (var61 * var147) - (var63 * var164)
    var166 = var69 * (var165 - ((var74 * var153) + (var68 * var161)))
    var167 = var159 + (var71 * var154)
    var168 = var69 * (var167 + ((var76 * var161) + (var74 * var162)))
    var169 = (var81 * var153) - (var79 * var161)
    var170 = (var79 * var153) + (var81 * var161)
    var171 = var84 * (var165 - ((var85 * var170) - (var83 * var169)))
    var172 = var167 + (var89 * var161)
    var173 = (var61 * var164) + (var63 * var147)
    var174 = var173 + (var89 * var153)
    var175 = var84 * (((var79 * var172) - (var81 * var174)) - ((var88 * var169) - (var85 * var162)))
    var176 = (var80 * var169) + ((var83 * var171) - (var88 * var175))
    var177 = var84 * (((var79 * var174) + (var81 * var172)) - ((var83 * var162) - (var88 * var170)))
    var178 = (var80 * var170) + ((var88 * var177) - (var85 * var171))
    var179 = (var81 * var177) + (var79 * var175)
    var180 = ((((var65 * var153) - (var62 * var161)) + (var66 * var162)) + ((var68 * var166) - (var76 * var168))) + (
        ((var79 * var176) - (var81 * var178)) - (var89 * var179)
    )
    var181 = var69 * (var173 - ((var68 * var162) - (var76 * var153)))
    var182 = (var102 * var162) + ((var85 * var175) - (var83 * var177))
    var183 = (
        (((var99 * var153) - (var66 * var161)) + (var100 * var162)) + ((var74 * var168) - (var68 * var181))
    ) + var182
    var184 = var168 + var179
    var185 = ((((var48 * var153) - (var45 * var154)) - (var50 * var151)) + ((var51 * var155) - (var55 * var160))) + (
        ((var61 * var180) - (var63 * var183)) - (var71 * var184)
    )
    var186 = var52 * (var163 + ((var51 * var151) + (var55 * var153)))
    var187 = (var79 * var177) - (var81 * var175)
    var188 = ((((var110 * var153) - (var65 * var161)) + (var99 * var162)) + ((var76 * var181) - (var74 * var166))) + (
        ((var81 * var176) + (var79 * var178)) + (var89 * var187)
    )
    var189 = var181 + var187
    var190 = var166 + var171
    var191 = (var61 * var189) - (var63 * var190)
    var192 = ((((var107 * var153) - (var48 * var154)) - (var108 * var151)) + ((var55 * var186) - (var53 * var155))) + (
        var188 + (var71 * var191)
    )
    var193 = var186 + var191
    var194 = var160 + var184
    var195 = (var46 * var193) + (var44 * var194)
    var196 = (((var33 * var149) - (var41 * var152)) - (((var27 * var148) + (var30 * var10)) + (var31 * var151))) + (
        ((var44 * var185) - (var46 * var192)) - (var22 * var195)
    )
    var197 = var34 * (var157 - ((var41 * var10) - (var33 * var151)))
    var198 = ((((var108 * var153) - (var50 * var154)) - (var124 * var151)) + ((var53 * var160) - (var51 * var186))) + (
        (var63 * var180) + (var61 * var183)
    )
    var199 = (
        ((var35 * var152) - (var33 * var197)) - (((var31 * var148) + (var121 * var10)) + (var122 * var151))
    ) + var198
    var200 = var152 + var195
    var201 = (((var17 * var141) - (var21 * var144)) - ((var11 * var14) + (var13 * var10))) + (
        ((var26 * var196) + (var28 * var199)) - (var37 * var200)
    )
    var202 = var18 * (var145 - (var21 * var10))
    var203 = (var44 * var193) - (var46 * var194)
    var204 = (((var41 * var197) - (var35 * var149)) - (((var30 * var148) + (var132 * var10)) + (var121 * var151))) + (
        ((var46 * var185) + (var44 * var192)) + (var22 * var203)
    )
    var205 = (var26 * (var197 + var203)) + (var28 * (var149 + (var155 + ((var63 * var189) + (var61 * var190)))))
    var206 = (((var21 * var202) - (var19 * var141)) - ((var13 * var14) + (var129 * var10))) + (
        var204 + (var37 * var205)
    )
    var207 = var202 + var205
    var208 = var144 + var200
    var209 = (((var19 * var144) - (var17 * var202)) - ((var16 * var14) + (var130 * var10))) + (
        (var26 * var199) - (var28 * var196)
    )
    var210 = var34 * (var33 * var28)
    var211 = var34 * ((var41 * var28) + (var35 * var26))
    var212 = var46 * var28
    var213 = var44 * var28
    var214 = var52 * ((var53 * var212) + (var51 * var213))
    var215 = var22 * var28
    var216 = var44 * var215
    var217 = var52 * (var216 + ((var55 * var213) + (var53 * var26)))
    var218 = (var63 * var26) - (var61 * var213)
    var219 = (var61 * var26) + (var63 * var213)
    var220 = var46 * var215
    var221 = var220 + (var71 * var212)
    var222 = var63 * var221
    var223 = var69 * (var222 + ((var74 * var212) - (var68 * var218)))
    var224 = var216 + (var71 * var213)
    var225 = var69 * (var224 - ((var76 * var218) - (var74 * var219)))
    var226 = (var79 * var218) + (var81 * var212)
    var227 = (var79 * var212) - (var81 * var218)
    var228 = var84 * (var222 + ((var85 * var227) - (var83 * var226)))
    var229 = var224 - (var89 * var218)
    var230 = var61 * var221
    var231 = var230 + (var89 * var212)
    var232 = var84 * (((var79 * var229) - (var81 * var231)) - ((var88 * var226) - (var85 * var219)))
    var233 = (var80 * var226) - ((var83 * var228) + (var88 * var232))
    var234 = var84 * (((var79 * var231) + (var81 * var229)) - ((var83 * var219) - (var88 * var227)))
    var235 = (var80 * var227) + ((var88 * var234) + (var85 * var228))
    var236 = (var81 * var234) + (var79 * var232)
    var237 = ((((var62 * var218) + (var65 * var212)) + (var66 * var219)) - ((var68 * var223) + (var76 * var225))) + (
        ((var79 * var233) - (var81 * var235)) - (var89 * var236)
    )
    var238 = var69 * (var230 - ((var68 * var219) - (var76 * var212)))
    var239 = (var102 * var219) + ((var85 * var232) - (var83 * var234))
    var240 = (
        (((var66 * var218) + (var99 * var212)) + (var100 * var219)) + ((var74 * var225) - (var68 * var238))
    ) + var239
    var241 = var225 + var236
    var242 = ((((var48 * var212) - (var45 * var213)) + (var50 * var26)) - ((var51 * var214) + (var55 * var217))) + (
        ((var61 * var237) - (var63 * var240)) - (var71 * var241)
    )
    var243 = var52 * (var220 - ((var51 * var26) - (var55 * var212)))
    var244 = (var79 * var234) - (var81 * var232)
    var245 = ((((var65 * var218) + (var110 * var212)) + (var99 * var219)) + ((var76 * var238) + (var74 * var223))) + (
        ((var81 * var233) + (var79 * var235)) + (var89 * var244)
    )
    var246 = var238 + var244
    var247 = var223 + var228
    var248 = (var61 * var246) + (var63 * var247)
    var249 = ((((var107 * var212) - (var48 * var213)) + (var108 * var26)) + ((var55 * var243) + (var53 * var214))) + (
        var245 + (var71 * var248)
    )
    var250 = var243 + var248
    var251 = var217 + var241
    var252 = (var46 * var250) + (var44 * var251)
    var253 = (((var31 * var26) - (var27 * var28)) - ((var33 * var210) + (var41 * var211))) + (
        ((var44 * var242) - (var46 * var249)) - (var22 * var252)
    )
    var254 = var34 * (var33 * var26)
    var255 = ((((var108 * var212) - (var50 * var213)) + (var124 * var26)) + ((var53 * var217) - (var51 * var243))) + (
        (var63 * var237) + (var61 * var240)
    )
    var256 = (((var122 * var26) - (var31 * var28)) + ((var35 * var211) + (var33 * var254))) + var255
    # var257 = var211 + var252
    # var258 = 5.0137e-07 + (((var26 * var253) + (var28 * var256)) - (var37 * var257))
    # var259 = (var44 * var250) - (var46 * var251)
    # var260 = (((var121 * var26) - (var30 * var28)) + ((var35 * var210) - (var41 * var254))) + (
    #    ((var46 * var242) + (var44 * var249)) + (var22 * var259)
    # )
    # var261 = (var26 * (var259 - var254)) + (var28 * ((((var63 * var246) - (var61 * var247)) - var214) - var210))
    # var262 = 0.00315704 + (var260 + (var37 * var261))
    # var263 = 0.0537383 + var261
    # var264 = -3.24601e-06 + var257
    var265 = 0.00566084 + ((var26 * var256) - (var28 * var253))
    var266 = var22 * var46
    var267 = var52 * (var266 + (var55 * var46))
    var268 = var52 * ((var53 * var44) - (var51 * var46))
    var269 = var61 * var46
    var270 = var63 * var46
    var271 = var266 + (var71 * var46)
    var272 = var69 * (var271 + ((var76 * var269) + (var74 * var270)))
    var273 = var22 * var44
    var274 = var273 + (var71 * var44)
    var275 = var63 * var274
    var276 = var69 * (var275 + ((var74 * var44) - (var68 * var269)))
    var277 = (var79 * var269) + (var81 * var44)
    var278 = var271 + (var89 * var269)
    var279 = var61 * var274
    var280 = var279 + (var89 * var44)
    var281 = var84 * (((var79 * var278) + (var81 * var280)) + ((var88 * var277) + (var85 * var270)))
    var282 = (var79 * var44) - (var81 * var269)
    var283 = var84 * (var275 + ((var85 * var282) - (var83 * var277)))
    var284 = (var80 * var277) + ((var88 * var281) - (var83 * var283))
    var285 = var84 * (((var79 * var280) - (var81 * var278)) + ((var83 * var270) + (var88 * var282)))
    var286 = (var80 * var282) + ((var88 * var285) + (var85 * var283))
    var287 = (var81 * var285) - (var79 * var281)
    var288 = ((((var62 * var269) + (var65 * var44)) - (var66 * var270)) + ((var76 * var272) - (var68 * var276))) + (
        ((var79 * var284) - (var81 * var286)) - (var89 * var287)
    )
    var289 = var69 * (var279 + ((var68 * var270) + (var76 * var44)))
    var290 = (var102 * var270) + ((var85 * var281) + (var83 * var285))
    var291 = (
        (((var66 * var269) + (var99 * var44)) - (var100 * var270)) - ((var74 * var272) + (var68 * var289))
    ) - var290
    var292 = var287 - var272
    var293 = (((var45 * var46) + (var48 * var44)) + ((var55 * var267) - (var51 * var268))) + (
        ((var61 * var288) - (var63 * var291)) - (var71 * var292)
    )
    var294 = var52 * (var273 + (var55 * var44))
    var295 = (var79 * var285) + (var81 * var281)
    var296 = ((((var65 * var269) + (var110 * var44)) - (var99 * var270)) + ((var76 * var289) + (var74 * var276))) + (
        ((var81 * var284) + (var79 * var286)) + (var89 * var295)
    )
    var297 = var289 + var295
    var298 = var276 + var283
    var299 = (var61 * var297) + (var63 * var298)
    var300 = (((var48 * var46) + (var107 * var44)) + ((var55 * var294) + (var53 * var268))) + (
        var296 + (var71 * var299)
    )
    var301 = var294 + var299
    var302 = var292 - var267
    var303 = (var46 * var301) + (var44 * var302)
    var304 = -6.73952e-08 + (((var44 * var293) - (var46 * var300)) - (var22 * var303))
    var305 = (((var50 * var46) + (var108 * var44)) - ((var53 * var267) + (var51 * var294))) + (
        (var63 * var288) + (var61 * var291)
    )
    var306 = -0.00783622 + var305
    # var307 = ((var26 * var304) + (var28 * var306)) - (var37 * var303)
    var308 = (var44 * var301) - (var46 * var302)
    var309 = 0.0266291 + (((var46 * var293) + (var44 * var300)) + (var22 * var308))
    # var310 = (var26 * (0.169668 + var308)) + (var28 * (-2.58536e-07 + (((var63 * var297) - (var61 * var298)) - var268)))
    # var311 = var309 + (var37 * var310)
    var312 = (var26 * var306) - (var28 * var304)
    var313 = var69 * (var68 * var63)
    var314 = var69 * ((var76 * var63) - (var74 * var61))
    var315 = var79 * var63
    var316 = var81 * var63
    var317 = var84 * ((var85 * var316) + (var83 * var315))
    var318 = var89 * var63
    var319 = var84 * ((var79 * var318) + ((var88 * var315) - (var85 * var61)))
    var320 = (var80 * var315) + ((var83 * var317) + (var88 * var319))
    var321 = var84 * ((var81 * var318) + ((var83 * var61) + (var88 * var316)))
    var322 = (var80 * var316) + ((var88 * var321) + (var85 * var317))
    var323 = (var81 * var321) + (var79 * var319)
    var324 = (((var62 * var63) + (var66 * var61)) + ((var68 * var313) + (var76 * var314))) + (
        ((var79 * var320) + (var81 * var322)) + (var89 * var323)
    )
    var325 = var69 * (var68 * var61)
    var326 = (var102 * var61) + ((var83 * var321) - (var85 * var319))
    var327 = (((var66 * var63) + (var100 * var61)) + ((var68 * var325) - (var74 * var314))) + var326
    var328 = var314 + var323
    var329 = 2.62175e-08 + (((var61 * var324) - (var63 * var327)) + (var71 * var328))
    var330 = (var81 * var319) - (var79 * var321)
    var331 = (((var65 * var63) + (var99 * var61)) - ((var76 * var325) + (var74 * var313))) + (
        ((var81 * var320) - (var79 * var322)) + (var89 * var330)
    )
    var332 = var330 - var325
    var333 = var313 + var317
    var334 = (var61 * var332) - (var63 * var333)
    var335 = -0.000252129 + (var331 + (var71 * var334))
    var336 = -0.0332527 + var334
    var337 = -9.61679e-07 - var328
    var338 = (var46 * var336) + (var44 * var337)
    var339 = ((var44 * var329) - (var46 * var335)) - (var22 * var338)
    var340 = 0.00428795 + ((var63 * var324) + (var61 * var327))
    # var341 = ((var26 * var339) + (var28 * var340)) - (var37 * var338)
    var342 = (var44 * var336) - (var46 * var337)
    var343 = ((var46 * var329) + (var44 * var335)) + (var22 * var342)
    # var344 = (var26 * var342) + (var28 * ((var63 * var332) + (var61 * var333)))
    # var345 = var343 + (var37 * var344)
    var346 = (var26 * var340) - (var28 * var339)
    var347 = var84 * ((var83 * var81) - (var85 * var79))
    var348 = -0.078
    var349 = var84 * ((var88 * var81) - (var348 * var81))
    var350 = (var80 * var81) + ((var83 * var347) + (var88 * var349))
    var351 = var84 * ((var348 * var79) - (var88 * var79))
    var352 = ((var88 * var351) + (var85 * var347)) - (var80 * var79)
    var353 = (var81 * var351) + (var79 * var349)
    var354 = 4.155e-09 - (((var79 * var350) + (var81 * var352)) + (var89 * var353))
    var355 = (var85 * var349) - (var83 * var351)
    var356 = 4.52838e-05 + var355
    var357 = ((var61 * var354) - (var63 * var356)) - (var71 * var353)
    var358 = (var79 * var351) - (var81 * var349)
    var359 = -0.0299965 + (((var79 * var352) - (var81 * var350)) + (var89 * var358))
    var360 = 0.00440927 + var358
    var361 = -3.94934e-08 - var347
    var362 = (var61 * var360) - (var63 * var361)
    var363 = var359 + (var71 * var362)
    var364 = (var46 * var362) + (var44 * var353)
    var365 = ((var44 * var357) - (var46 * var363)) - (var22 * var364)
    var366 = (var63 * var354) + (var61 * var356)
    # var367 = ((var26 * var365) + (var28 * var366)) - (var37 * var364)
    var368 = (var44 * var362) - (var46 * var353)
    var369 = ((var46 * var357) + (var44 * var363)) + (var22 * var368)
    # var370 = (var26 * var368) + (var28 * ((var63 * var360) + (var61 * var361)))
    # var371 = var369 + (var37 * var370)
    var372 = (var26 * var366) - (var28 * var365)
    var373 = 4.77082e-20
    var374 = 1.90833e-19
    var375 = -1.20668e-17
    var376 = 3.0167e-18
    var377 = (var375 * var81) + (var376 * var79)
    var378 = ((var373 * var79) - (var374 * var81)) - (var89 * var377)
    var379 = ((var61 * var378) - (var102 * var63)) - (var71 * var377)
    var380 = (var375 * var79) - (var376 * var81)
    var381 = ((var373 * var81) + (var374 * var79)) + (var89 * var380)
    var382 = var61 * var380
    var383 = var381 + (var71 * var382)
    var384 = (var46 * var382) + (var44 * var377)
    var385 = ((var44 * var379) - (var46 * var383)) - (var22 * var384)
    var386 = (var63 * var378) + (var102 * var61)
    # var387 = ((var26 * var385) + (var28 * var386)) - (var37 * var384)
    var388 = (var44 * var382) - (var46 * var377)
    var389 = ((var46 * var379) + (var44 * var383)) + (var22 * var388)
    # var390 = (var26 * var388) + (var28 * (var63 * var380))
    # var391 = var389 + (var37 * var390)
    var392 = (var26 * var386) - (var28 * var385)

    Mq[:, 0, 0] = 0.045327 + (
        (
            var1
            * (
                (((0.0142349 * var1) + (var2 * var3)) + ((var4 * var6) + (var7 * var9)))
                + (((var10 * var128) - (var14 * var136)) - (var22 * ((var14 * var137) + (var10 * var138))))
            )
        )
        + (var3 * ((((var2 * var1) + (0.00424792 * var3)) - ((var8 * var9) - (var4 * var139))) + var140))
    )

    Mq[:, 0, 1] = Mq[:, 1, 0] = (
        var1
        * (-5.82493e-08 + (((var10 * var201) - (var14 * var206)) - (var22 * ((var14 * var207) + (var10 * var208)))))
    ) + (var3 * (-0.00783671 + var209))

    # Mq[:, 0, 2] = Mq[:, 2, 0] = (
    #     var1 * (((var10 * var258) - (var14 * var262)) - (var22 * ((var14 * var263) + (var10 * var264))))
    # ) + (var3 * var265)

    # Mq[:, 0, 3] = Mq[:, 3, 0] = (
    #     var1 * (((var10 * var307) - (var14 * var311)) - (var22 * ((var14 * var310) + (var10 * var303))))
    # ) + (var3 * var312)

    # Mq[:, 0, 4] = Mq[:, 4, 0] = (
    #     var1 * (((var10 * var341) - (var14 * var345)) - (var22 * ((var14 * var344) + (var10 * var338))))
    # ) + (var3 * var346)

    # Mq[:, 0, 5] = Mq[:, 5, 0] = (
    #     var1 * (((var10 * var367) - (var14 * var371)) - (var22 * ((var14 * var370) + (var10 * var364))))
    # ) + (var3 * var372)

    # Mq[:, 0, 6] = Mq[:, 6, 0] = (
    #     var1 * (((var10 * var387) - (var14 * var391)) - (var22 * ((var14 * var390) + (var10 * var384))))
    # ) + (var3 * var392)

    # Mq[:, 1, 0] = -(
    #     (((-3.73763e-08 * var1) + (0.0022809 * var3)) - ((var7 * var139) + (var8 * var6)))
    #     + (((var14 * var128) + (var10 * var136)) + (var22 * ((var10 * var137) - (var14 * var138))))
    # )

    Mq[:, 1, 1] = -(
        -0.0266285 + (((var14 * var201) + (var10 * var206)) + (var22 * ((var10 * var207) - (var14 * var208))))
    )

    # Mq[:, 1, 2] = Mq[:, 2, 1] = -(
    #     ((var14 * var258) + (var10 * var262)) + (var22 * ((var10 * var263) - (var14 * var264)))
    # )

    # Mq[:, 1, 3] = Mq[:, 3, 1] = -(
    #     ((var14 * var307) + (var10 * var311)) + (var22 * ((var10 * var310) - (var14 * var303)))
    # )

    # Mq[:, 1, 4] = Mq[:, 4, 1] = -(
    #     ((var14 * var341) + (var10 * var345)) + (var22 * ((var10 * var344) - (var14 * var338)))
    # )

    # Mq[:, 1, 5] = Mq[:, 5, 1] = -(
    #     ((var14 * var367) + (var10 * var371)) + (var22 * ((var10 * var370) - (var14 * var364)))
    # )

    # Mq[:, 1, 6] = Mq[:, 6, 1] = -(
    #     ((var14 * var387) + (var10 * var391)) + (var22 * ((var10 * var390) - (var14 * var384)))
    # )

    Mq[:, 2, 0] = Mq[:, 0, 2] = var140

    Mq[:, 2, 1] = Mq[:, 1, 2] = var209

    Mq[:, 2, 2] = var265

    Mq[:, 2, 3] = Mq[:, 3, 2] = var312

    Mq[:, 2, 4] = Mq[:, 4, 2] = var346

    Mq[:, 2, 5] = Mq[:, 5, 2] = var372

    Mq[:, 2, 6] = Mq[:, 6, 2] = var392

    Mq[:, 3, 0] = Mq[:, 0, 3] = var134

    Mq[:, 3, 1] = Mq[:, 1, 3] = var204

    # Mq[:, 3, 2] = var260

    Mq[:, 3, 3] = var309

    Mq[:, 3, 4] = Mq[:, 4, 3] = var343

    Mq[:, 3, 5] = Mq[:, 5, 3] = var369

    Mq[:, 3, 6] = Mq[:, 6, 3] = var389

    Mq[:, 4, 0] = Mq[:, 0, 4] = var125

    Mq[:, 4, 1] = Mq[:, 1, 4] = var198

    # Mq[:, 4, 2] = var255

    # Mq[:, 4, 3] = var305

    Mq[:, 4, 4] = var340

    Mq[:, 4, 5] = Mq[:, 5, 4] = var366

    Mq[:, 4, 6] = Mq[:, 6, 4] = var386

    Mq[:, 5, 0] = Mq[:, 0, 5] = -var112

    Mq[:, 5, 1] = Mq[:, 1, 5] = -var188

    # Mq[:, 5, 2] = -var245

    # Mq[:, 5, 3] = -var296

    # Mq[:, 5, 4] = -var331

    Mq[:, 5, 5] = -var359

    Mq[:, 5, 6] = Mq[:, 6, 5] = -var381

    Mq[:, 6, 0] = Mq[:, 0, 6] = var103

    Mq[:, 6, 1] = Mq[:, 1, 6] = var182

    # Mq[:, 6, 2] = var239

    # Mq[:, 6, 3] = -var290

    # Mq[:, 6, 4] = Mq[:, 4, 6] = var326

    # Mq[:, 6, 5] = var355

    Mq[:, 6, 6] = var102

    return Mq
