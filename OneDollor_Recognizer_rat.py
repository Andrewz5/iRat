import cv2
import numpy as np
from dollarpy import Recognizer, Template, Point


def recognize(points):
    z=[]

    tmpl_1 = Template('incursion',[Point(175, 359), Point(177, 379), Point(184, 395), Point(197, 398), Point(222, 395), Point(238, 381), Point(258, 360), Point(278, 335), Point(301, 305), Point(323, 279), Point(336, 256), Point(354, 237), Point(361, 226), Point(386, 188), Point(392, 177), Point(400, 169), Point(405, 160)])
    tmpl_2 = Template('scaning',[Point(585, 98), Point(576, 104), Point(572, 113), Point(557, 127), Point(542, 140), Point(528, 156), Point(501, 182), Point(482, 198), Point(455, 215), Point(448, 216), Point(431, 213), Point(432, 208), Point(441, 178), Point(449, 161), Point(470, 139), Point(484, 117), Point(494, 110), Point(506, 96), Point(520, 87), Point(532, 78), Point(537, 61)])
    tmpl_3 = Template('focuedSearch',[Point(494, 187), Point(487, 219), Point(449, 237), Point(453, 186), Point(506, 186), Point(496, 259), Point(473, 213), Point(481, 163), Point(542, 248), Point(472, 254), Point(436, 184), Point(534, 184), Point(462, 224)])
    tmpl_4 = Template('chainingResponse',[Point(88, 232), Point(106, 276), Point(123, 295), Point(132, 313), Point(148, 328), Point(179, 340), Point(189, 339), Point(224, 337), Point(246, 332), Point(267, 322), Point(279, 306), Point(288, 289), Point(297, 271), Point(304, 248), Point(311, 223), Point(315, 211)])
    tmpl_5=Template('chainingResponse',[Point(511, 270), Point(477, 287), Point(441, 300), Point(414, 300), Point(327, 284), Point(311, 254), Point(309, 223), Point(309, 210)])

    tmpl_6 = Template('selfOrienting',[Point(565, 165), Point(556, 185), Point(540, 193), Point(509, 193), Point(471, 193), Point(449, 171), Point(453, 145), Point(473, 127), Point(484, 144), Point(490, 177), Point(488, 197), Point(483, 212), Point(473, 225), Point(440, 236), Point(417, 244)])
    tmpl_7 = Template('incursion',[Point(329, 81), Point(308, 67), Point(278, 51), Point(253, 72), Point(254, 101), Point(258, 135), Point(255, 162), Point(257, 202), Point(257, 237), Point(256, 270), Point(251, 296)])
    tmpl_8 = Template('selfOrienting',[Point(548, 173), Point(536, 183), Point(518, 186), Point(500, 181), Point(495, 161), Point(503, 170), Point(503, 174), Point(500, 190), Point(485, 232)])
    




    # tmpl_1 = Template('incursion',[Point(238, 309), Point(268, 341), Point(308, 332), Point(344, 292), Point(390, 217), Point(413, 167), Point(455, 112), Point(480, 80), Point(490, 69)])
    # tmpl_2 = Template('scaning',[Point(590, 73), Point(577, 95), Point(556, 132), Point(522, 176), Point(505, 206), Point(466, 252), Point(435, 285), Point(395, 283), Point(396, 259), Point(415, 186),Point(442, 140), Point(459, 117), Point(461, 87), Point(457, 54)])
    # tmpl_3 = Template('focuedSearch',[Point(449, 280), Point(457, 295), Point(430, 291), Point(450, 276), Point(486, 289), Point(436, 317), Point(431, 276), Point(502, 266), Point(467, 306), Point(399, 267), Point(451, 261), Point(491, 319), Point(389, 305), Point(475, 256)])
    # tmpl_4 = Template('chainingResponse',[Point(141, 158), Point(144, 197), Point(151, 237), Point(152, 263), Point(160, 292), Point(172, 321), Point(188, 341), Point(206, 348), Point(246, 353), Point(283, 355), Point(323, 351), Point(344, 330), Point(353, 308), Point(365, 279), Point(368, 242), Point(365, 209), Point(366, 175), Point(366, 159)])
    # tmpl_5 = Template('selfOrienting',[Point(502, 238), Point(486, 256), Point(456, 266), Point(429, 262), Point(424, 243), Point(435, 228), Point(452, 241), Point(455, 252), Point(455, 267), Point(455, 282), Point(451, 305)])
    

    # ===================================================================================================================
    ### OLd Points
    # ===================================================================================================================
    
    # tmpl_1 = Template('Chaining Response',[Point(343, 140), Point(379, 189), Point(400, 260), Point(391, 306), Point(361, 319), Point(342, 312),
    # Point(291, 289), Point(285, 281), Point(248, 256), Point(238, 250),Point(211, 221), Point(197, 197)])

    # tmpl_2 = Template('thigmotaxis',[Point(619, 187), Point(617, 195), Point(614, 206), Point(613, 216), Point(611, 220), Point(605, 230), Point(591, 246),
    # Point(586, 256), Point(577, 267), Point(572, 272), Point(557, 284), Point(547, 287), Point(537, 295), Point(520, 304),
    # Point(507, 312), Point(487, 319), Point(467, 321), Point(440, 326), Point(417, 322), Point(395, 326), Point(373, 328),
    # Point(355, 325), Point(355, 325), Point(325, 320), Point(308, 315), Point(270, 298), Point(250, 283),Point(240, 265),
    # Point(226, 234)])
    
    # tmpl_3 = Template('incursion',[Point(227, 121), Point(219, 145), Point(202, 180), 
    # Point(185, 213), Point(174, 238), Point(160, 258), Point(129, 238), Point(112, 202)])

    # tmpl_4 = Template('scanning',[Point(475, 132), Point(464, 145), Point(446, 164), Point(430, 189), Point(412, 210),
    # Point(390, 228), Point(372, 239), Point(356, 253), Point(343, 237), Point(355, 210), Point(369, 195), Point(379, 171),
    # Point(392, 159), Point(409, 139), Point(417, 130), Point(440, 111)])

    # tmpl_5 = Template('self orienting',[Point(557, 173), Point(540, 184), Point(519, 191), Point(491, 195),
    # Point(469, 177), Point(485, 152), Point(512, 149), Point(527, 162), Point(534, 183), Point(542, 201), Point(550, 222)])
    
    
    recognizer = Recognizer([tmpl_1,tmpl_2,tmpl_3,tmpl_4,tmpl_5,tmpl_6,tmpl_7,tmpl_8])
    for pt in points:
        x,y=pt
        z.append(Point(x,y))

    # Call 'recognize(...)' to match a list of 'Point' elements to the previously defined templates.
    result = recognizer.recognize(z)
    a,b=result
    if (a!=None):
        print("1$ ==> ",result)  # Output: ('X', 0.733770116545184)

    return result


        


