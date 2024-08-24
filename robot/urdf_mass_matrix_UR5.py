import torch
import torch.jit as jit


def compute_mass_matrix_urdf(q):

    horizon = q.shape[1]
    Mq = torch.empty(horizon, 6, 6, dtype=torch.float32).to("cuda")
    q1 = q[0, :]
    q2 = q[1, :]
    q3 = q[2, :]
    q4 = q[3, :]
    q5 = q[4, :]
    q6 = q[5, :]

    c1, s1 = torch.cos(q1), torch.sin(q1)
    c2, s2 = torch.cos(q2), torch.sin(q2)
    c3, s3 = torch.cos(q3), torch.sin(q3)
    c4, s4 = torch.cos(q4), torch.sin(q4)
    c5, s5 = torch.cos(q5), torch.sin(q5)
    # c6, s6 = torch.cos(q6), torch.sin(q6)
    # c7, s7 = torch.cos(q7), torch.sin(q7)

    var1 = s1
    var2 = 0.136
    var3 = 8.393
    var4 = var3 * ((var2 * var1) + -4.35845e-11)
    var5 = c2
    var6 = s2
    var7 = c1
    var8 = (var5 * var1) + (var6 * var7)
    var9 = 0.0165
    var10 = 2.275
    var11 = 8.7169e-11
    var12 = var11 * var5
    var13 = var10 * (var12 - ((var9 * var8) + -4.02259e-11))
    var14 = c3
    var15 = 0.00201389
    var16 = s3
    var17 = (var5 * var7) - (var6 * var1)
    var18 = (var14 * var8) + (var16 * var17)
    var19 = c4
    var20 = 0.00183104
    var21 = -2.05103e-10
    var22 = s4
    var23 = var21 * var22
    var24 = (var14 * var17) - (var16 * var8)
    var25 = (var19 * var18) + ((var23 * var24) + (var21 * var22))
    var26 = c5

    var28 = s5
    var29 = var26
    var30 = 8.06247e-05

    var32 = var21 * var28
    var33 = var21 * var19
    var34 = ((var33 * var24) + (var21 * var19)) - (var22 * var18)
    var35 = -var28

    var37 = -var24
    var38 = (var29 * var25) + ((var32 * var34) + (var35 * var37))
    var39 = -0.01525
    var40 = 0.1879
    var41 = -var28
    var42 = var11 * var6
    var43 = 0.10915
    var44 = var42 + (var43 * var17)
    var45 = var12 - ((var43 * var8) + -8.04518e-11)
    var46 = (var14 * var44) + (var16 * var45)
    var47 = -1.9413e-11
    var48 = var46 - (1.9413e-11 - (var47 * var24))
    var49 = (var14 * var45) - (var16 * var44)
    var50 = var49 - (var47 * var18)
    var51 = -0.09465
    var52 = -0.425
    var53 = var52 * var7
    var54 = -0.39225
    var55 = var53 + (var54 * var17)
    var56 = (var51 * var18) - var55
    var57 = (var19 * var48) + ((var23 * var50) + (var22 * var56))
    var58 = 0.0823
    var59 = -1.688e-11
    var60 = var57 - ((var58 * var37) - (var59 * var34))
    var61 = var21 * var26
    var62 = ((var33 * var50) + (var19 * var56)) - (var22 * var48)
    var63 = var62 - (var59 * var25)
    var64 = var26
    var65 = (var21 * var56) - var50
    var66 = var65 + (var58 * var25)
    var67 = var40 * (((var41 * var60) + ((var61 * var63) - (var64 * var66))) - (var39 * var38))
    var68 = (var30 * var38) - (var39 * var67)
    var69 = (var41 * var25) + ((var61 * var34) - (var64 * var37))
    var70 = var40 * (((var29 * var60) + ((var32 * var63) + (var35 * var66))) + (var39 * var69))
    var71 = (var30 * var69) + (var39 * var70)

    var73 = 0.000132117
    var74 = var73 * ((var34 + (var21 * var37)))
    var75 = var40 * ((var63 + (var21 * var66)))
    var76 = (var35 * var70) + ((var21 * var75) - (var64 * var67))
    var77 = (var32 * var70) + ((var61 * var67) + var75)
    var78 = (var20 * var25) + (((var29 * var68) + ((var41 * var71))) + ((var58 * var76) - (var59 * var77)))
    var79 = (var29 * var70) + ((var41 * var67))
    var80 = (var20 * var34) + (((var32 * var68) + ((var61 * var71) + var74)) + (var59 * var79))
    var81 = 1.219
    var82 = (var81 * var57) + var79
    var83 = (var81 * var62) + var77
    var84 = (var81 * var65) + var76
    var85 = (var22 * var82) + ((var19 * var83) + (var21 * var84))
    var86 = (var23 * var82) + ((var33 * var83) - var84)
    var87 = (var15 * var18) + (((var19 * var78) - (var22 * var80)) + ((var51 * var85) - (var47 * var86)))
    var88 = 0.0021942
    var89 = (var88 * var37) + (((var35 * var68) + ((var21 * var74) - (var64 * var71))) - (var58 * var79))
    var90 = (var19 * var82) - (var22 * var83)
    var91 = (var15 * var24) + (((var23 * var78) + ((var33 * var80) - var89)) + (var47 * var90))
    var92 = (var81 * var46) + var90
    var93 = (var81 * var49) + var86
    var94 = (var16 * var92) + (var14 * var93)
    var95 = (((0.004095 * var8)) - (var9 * var13)) + (((var14 * var87) - (var16 * var91)) - (var43 * var94))
    var96 = 0.0312168
    var97 = var10 * (var42 + (var9 * var17))
    var98 = -0.196125
    var99 = var10 * (var53 + (var98 * var17))
    var100 = (var14 * var92) - (var16 * var93)
    var101 = var85 - (var81 * var55)
    var102 = ((var96 * var17) + ((var9 * var97) + (var98 * var99))) + (
        ((var16 * var87) + (var14 * var91)) + ((var43 * var100) - (var54 * var101))
    )
    var103 = -0.2125
    var105 = -4.50038e-13
    var106 = var105 + (((var22 * var78) + ((var19 * var80) + (var21 * var89))) - (var51 * var90))
    var107 = (-6.40267e-12) + (var98 * var13) + var106 + (var54 * var94)
    var108 = (((-2.63741e-17 * var1) + -2.74604e-11) - (var103 * var4)) + (
        var107 + (var52 * ((var6 * (var97 + var100)) + (var5 * (var13 + var94))))
    )
    var109 = var52 * var5
    var110 = var10 * (var109 + var98)
    var111 = (var29 * var22) + ((var32 * var19) + (var21 * var35))
    var112 = var52 * var6
    var113 = var109 + var54
    var114 = (var14 * var112) + (var16 * var113)
    var115 = 0.09465
    var116 = var114 + var115
    var117 = (var14 * var113) - (var16 * var112)
    var118 = (var19 * var116) + (var23 * var117)
    var119 = -1.688e-11
    var120 = var118 - (var119 - (var59 * var19))
    var121 = (var33 * var117) - (var22 * var116)
    var122 = var121 - (var59 * var22)
    var123 = (var58 * var22) - var117
    var124 = var40 * (((var41 * var120) + ((var61 * var122) - (var64 * var123))) - (var39 * var111))
    var125 = (var30 * var111) - (var39 * var124)
    var126 = (var41 * var22) + ((var61 * var19) - (var21 * var64))
    var127 = var40 * (((var29 * var120) + ((var32 * var122) + (var35 * var123))) + (var39 * var126))
    var128 = (var30 * var126) + (var39 * var127)
    var129 = var73 * (var19)
    var130 = var40 * ((var122 + (var21 * var123)))
    var131 = (var35 * var127) + ((var21 * var130) - (var64 * var124))
    var132 = (var32 * var127) + ((var61 * var124) + var130)
    var133 = (var20 * var22) + (((var29 * var125) + ((var41 * var128))) + ((var58 * var131) - (var59 * var132)))
    var134 = (var29 * var127) + ((var41 * var124))
    var135 = (var20 * var19) + (((var32 * var125) + ((var61 * var128) + var129)) + (var59 * var134))
    var136 = (var81 * var118) + var134
    var137 = (var81 * var121) + var132
    var138 = var131 - (var81 * var117)
    var139 = (var22 * var136) + ((var19 * var137) + (var21 * var138))
    var140 = (var23 * var136) + ((var33 * var137) - var138)
    var141 = ((var19 * var133) - (var22 * var135)) + ((var51 * var139) - (var47 * var140))
    var142 = var105 + (((var35 * var125) + ((var21 * var129) - (var64 * var128))) - (var58 * var134))
    var143 = (var19 * var136) - (var22 * var137)
    var144 = ((var23 * var133) + ((var33 * var135) - var142)) + (var47 * var143)
    var145 = (var81 * var114) + var143
    var146 = (var81 * var117) + var140
    var147 = (var16 * var145) + (var14 * var146)
    var148 = (-(var9 * var110)) + (((var14 * var141) - (var16 * var144)) - (var43 * var147))
    var149 = var10 * var112
    var150 = (var14 * var145) - (var16 * var146)
    var151 = (var9 * var149) + (((var16 * var141) + (var14 * var144)) + ((var43 * var150) - (var54 * var139)))
    var152 = var88 + (((var22 * var133) + ((var19 * var135) + (var21 * var142))) - (var51 * var143))
    var153 = (var96 + (var98 * var110)) + (var152 + (var54 * var147))
    var154 = 0.512882 + (var153 + (var52 * ((var6 * (var149 + var150)) + (var5 * (var110 + var147)))))
    var155 = (var29 * var22) + ((var32 * var19) + (var21 * var35))
    var156 = var54 * var16
    var157 = var156 + var115
    var158 = var54 * var14
    var159 = (var19 * var157) + (var23 * var158)
    var160 = var159 - (var119 - (var59 * var19))
    var161 = (var33 * var158) - (var22 * var157)
    var162 = var161 - (var59 * var22)
    var163 = (var58 * var22) - var158
    var164 = var40 * (((var41 * var160) + ((var61 * var162) - (var64 * var163))) - (var39 * var155))
    var165 = (var30 * var155) - (var39 * var164)
    var166 = (var41 * var22) + ((var61 * var19) - (var21 * var64))
    var167 = var40 * (((var29 * var160) + ((var32 * var162) + (var35 * var163))) + (var39 * var166))
    var168 = (var30 * var166) + (var39 * var167)
    var169 = var73 * (var19)
    var170 = var40 * ((var162 + (var21 * var163)))
    var171 = (var35 * var167) + ((var21 * var170) - (var64 * var164))
    var172 = (var32 * var167) + ((var61 * var164) + var170)
    var173 = (var20 * var22) + (((var29 * var165) + ((var41 * var168))) + ((var58 * var171) - (var59 * var172)))
    var174 = (var29 * var167) + ((var41 * var164))
    var175 = (var20 * var19) + (((var32 * var165) + ((var61 * var168) + var169)) + (var59 * var174))
    var176 = (var81 * var159) + var174
    var177 = (var81 * var161) + var172
    var178 = var171 - (var81 * var158)
    var179 = (var22 * var176) + ((var19 * var177) + (var21 * var178))
    var180 = (var23 * var176) + ((var33 * var177) - var178)
    var181 = ((var19 * var173) - (var22 * var175)) + ((var51 * var179) - (var47 * var180))
    var182 = var105 + (((var35 * var165) + ((var21 * var169) - (var64 * var168))) - (var58 * var174))
    var183 = (var19 * var176) - (var22 * var177)
    var184 = ((var23 * var173) + ((var33 * var175) - var182)) + (var47 * var183)
    var185 = (var81 * var156) + var183
    var186 = (var81 * var158) + var180
    var187 = (var16 * var185) + (var14 * var186)
    var188 = 0.00736204 + (((var14 * var181) - (var16 * var184)) - (var43 * var187))
    var189 = (var14 * var185) - (var16 * var186)
    var190 = ((var16 * var181) + (var14 * var184)) + ((var43 * var189) - (var54 * var179))
    var191 = var88 + (((var22 * var173) + ((var19 * var175) + (var21 * var182))) - (var51 * var183))
    var192 = 0.118725 + (var191 + (var54 * var187))
    var193 = var192 + (var52 * ((var6 * var189) + (var5 * (-0.446184 + var187))))
    var194 = (var29 * var22) + ((var32 * var19) + (var21 * var35))
    var195 = var115 * var19
    var196 = var195 - (var119 - (var59 * var19))
    var197 = var115 * var22
    var198 = var197 + (var59 * var22)
    var199 = var58 * var22
    var200 = var40 * (((var41 * var196) - ((var61 * var198) + (var64 * var199))) - (var39 * var194))
    var201 = (var30 * var194) - (var39 * var200)
    var202 = (var41 * var22) + ((var61 * var19) - (var21 * var64))
    var203 = var40 * (((var29 * var196) + ((var35 * var199) - (var32 * var198))) + (var39 * var202))
    var204 = (var30 * var202) + (var39 * var203)
    var205 = var73 * (var19)
    var206 = var40 * (((var21 * var199) - var198))
    var207 = (var35 * var203) + ((var21 * var206) - (var64 * var200))
    var208 = (var32 * var203) + ((var61 * var200) + var206)
    var209 = (var20 * var22) + (((var29 * var201) + ((var41 * var204))) + ((var58 * var207) - (var59 * var208)))
    var210 = (var29 * var203) + ((var41 * var200))
    var211 = (var20 * var19) + (((var32 * var201) + ((var61 * var204) + var205)) + (var59 * var210))
    var212 = (var81 * var195) + var210
    var213 = var208 - (var81 * var197)
    var214 = (var22 * var212) + ((var19 * var213) + (var21 * var207))
    var215 = (var23 * var212) + ((var33 * var213) - var207)
    var216 = ((var19 * var209) - (var22 * var211)) + ((var51 * var214) - (var47 * var215))
    var217 = var105 + (((var35 * var201) + ((var21 * var205) - (var64 * var204))) - (var58 * var210))
    var218 = (var19 * var212) - (var22 * var213)
    var219 = ((var23 * var209) + ((var33 * var211) - var217)) + (var47 * var218)
    var220 = (var16 * var218) + (var14 * var215)
    var221 = ((var14 * var216) - (var16 * var219)) - (var43 * var220)
    var222 = (var14 * var218) - (var16 * var215)
    var223 = ((var16 * var216) + (var14 * var219)) + ((var43 * var222) - (var54 * var214))
    var224 = var88 + (((var22 * var209) + ((var19 * var211) + (var21 * var217))) - (var51 * var218))
    var225 = var224 + (var54 * var220)
    var226 = var225 + (var52 * ((var6 * var222) + (var5 * var220)))
    var227 = -0.0823
    var228 = var40 * ((var227 * var41) - (var39 * var35))
    var229 = (var30 * var35) - (var39 * var228)
    var230 = var40 * ((var227 * var29) - (var39 * var64))
    var231 = (var39 * var230) - (var30 * var64)
    var232 = (var35 * var230) - (var64 * var228)
    var233 = (var32 * var230) + ((var61 * var228))
    var234 = ((var29 * var229) + ((var41 * var231))) + ((var58 * var232) - (var59 * var233))
    var235 = 0.0
    var236 = (var29 * var230) + ((var41 * var228))
    var237 = ((var32 * var229) + ((var61 * var231))) + (var59 * var236)
    var238 = (var22 * var236) + ((var19 * var233) + (var21 * var232))
    var239 = (var23 * var236) + ((var33 * var233) - var232)
    var240 = ((var19 * var234) - (var22 * var237)) + ((var51 * var238) - (var47 * var239))

    var242 = var88 + (((var35 * var229) + (-(var64 * var231))) - (var58 * var236))
    var243 = (var19 * var236) - (var22 * var233)
    var244 = ((var23 * var234) + ((var33 * var237) - var242)) + (var47 * var243)
    var245 = (var16 * var243) + (var14 * var239)
    var246 = ((var14 * var240) - (var16 * var244)) - (var43 * var245)
    var247 = (var14 * var243) - (var16 * var239)
    var248 = ((var16 * var240) + (var14 * var244)) + ((var43 * var247) - (var54 * var238))
    var249 = ((var22 * var234) + ((var19 * var237) + (var21 * var242))) - (var51 * var243)
    var250 = var249 + (var54 * var245)
    var251 = var250 + (var52 * ((var6 * var247) + (var5 * var245)))

    var253 = var73 * var22
    var254 = (var73 * var33) + 2.70977e-14
    var255 = (var14 * var253) - (var16 * var254)
    var256 = (var16 * var253) + (var14 * var254)
    var257 = var73 * var19

    # [
    #     [      00,       01,       02,       03,     04,     05]
    #     ["var108", "var154", "var193", "var226",  "var251", "var257"],
    #     ["var107", "var153", "var192", "var225",  "var250", "var257"],
    #     ["var106", "var152",  "var191", "var224", "var249", "var257"],
    #     ["var89",  "var142",   "var182", "var217","var242", "var235"],
    #     ["var74",  "var129",   "var169", "var205", "var235", "var73"],
    # ]

    Mq[:, 0, 0] = 0.0104063 + (
        (var1 * ((((0.0151074 * var1) + 5.40942e-27) + (var2 * var4)) + ((var5 * var95) - (var6 * var102))))
        + (
            (
                var7
                * (
                    ((0.133886 * var7) + ((var2 * (var3 * (var2 * var7))) + (var103 * (var3 * (var103 * var7)))))
                    + (((var6 * var95) + (var5 * var102)) - (var52 * (var101 - var99)))
                )
            )
            + (var21 * var108)
        )
    )

    Mq[:, 0, 1] = Mq[:, 1, 0] = (var1 * (0.242558 + ((var5 * var148) - (var6 * var151)))) + (
        (var7 * (((var6 * var148) + (var5 * var151)) - (var52 * var139))) + (var21 * var154)
    )

    Mq[:, 1, 1] = var154

    Mq[:, 2, 0] = Mq[:, 0, 2] = (var1 * ((var5 * var188) - (var6 * var190))) + (
        (var7 * (((var6 * var188) + (var5 * var190)) - (var52 * var179))) + (var21 * var193)
    )

    Mq[:, 2, 1] = Mq[:, 1, 2] = var153

    Mq[:, 2, 2] = var192

    Mq[:, 2, 3] = Mq[:, 3, 2] = var191

    Mq[:, 2, 4] = Mq[:, 4, 2] = var182

    Mq[:, 2, 5] = Mq[:, 5, 2] = var169

    Mq[:, 3, 0] = Mq[:, 0, 3] = (var1 * ((var5 * var221) - (var6 * var223))) + (
        (var7 * (((var6 * var221) + (var5 * var223)) - (var52 * var214))) + (var21 * var226)
    )

    Mq[:, 3, 1] = Mq[:, 1, 3] = var152

    Mq[:, 3, 3] = var224

    Mq[:, 3, 4] = Mq[:, 4, 3] = var217

    Mq[:, 3, 5] = Mq[:, 5, 3] = var205

    Mq[:, 4, 0] = Mq[:, 0, 4] = (var1 * ((var5 * var246) - (var6 * var248))) + (
        (var7 * (((var6 * var246) + (var5 * var248)) - (var52 * var238))) + (var21 * var251)
    )

    Mq[:, 4, 1] = Mq[:, 1, 4] = var142

    Mq[:, 4, 4] = var242

    Mq[:, 4, 5] = Mq[:, 5, 4] = var235

    Mq[:, 5, 0] = Mq[:, 0, 5] = (var1 * ((var5 * var255) - (var6 * var256))) + (
        (var7 * ((var6 * var255) + (var5 * var256))) + (var21 * var257)
    )

    Mq[:, 5, 1] = Mq[:, 1, 5] = var129

    Mq[:, 5, 5] = var73

    return Mq
