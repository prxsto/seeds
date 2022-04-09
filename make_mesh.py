import plotly.graph_objects as go
from math import sqrt

def midpoint(p1, p2):
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    z = (p1[2] + p2[2]) / 2
    return [x, y, z]

def make_window(pts, w, h, wwr, typology):

    wall_area = w * h
    glaz_area = wall_area * wwr
    win_width = w * .95
    win_height = glaz_area / win_width
    h_gap = (w - win_width) / 2.
    v_gap = (h - win_height) / 2.

    if typology == 1:
        
        # south
        n0 = [pts[0][0] + h_gap, pts[0][1] - .1, pts[0][2] + v_gap]
        n1 = [pts[1][0] - h_gap, pts[1][1] - .1, pts[1][2] + v_gap]
        n2 = [pts[2][0] - h_gap, pts[2][1] - .1, pts[2][2] - v_gap]
        n3 = [pts[3][0] + h_gap, pts[3][1] - .1, pts[3][2] - v_gap]
        
        # north
        n4 = [pts[4][0] + h_gap, pts[4][1] + .1, pts[4][2] + v_gap]
        n5 = [pts[5][0] - h_gap, pts[5][1] + .1, pts[5][2] + v_gap]
        n6 = [pts[6][0] - h_gap, pts[6][1] + .1, pts[6][2] - v_gap]
        n7 = [pts[7][0] + h_gap, pts[7][1] + .1, pts[7][2] - v_gap]
        
        # east
        n8 = [pts[1][0] + .1, pts[1][1] + h_gap, pts[1][2] + v_gap]
        n9 = [pts[5][0] + .1, pts[5][1] - h_gap, pts[5][2] + v_gap]
        n10 = [pts[6][0] + .1, pts[6][1] - h_gap, pts[6][2] - v_gap]
        n11 = [pts[2][0] + .1, pts[2][1] + h_gap, pts[2][2] - v_gap]
        
        # west
        n12 = [pts[0][0] - .1, pts[0][1] + h_gap, pts[0][2] + v_gap]
        n13 = [pts[4][0] - .1, pts[4][1] - h_gap, pts[4][2] + v_gap]
        n14 = [pts[7][0] - .1, pts[7][1] - h_gap, pts[7][2] - v_gap]
        n15 = [pts[3][0] - .1, pts[3][1] + h_gap, pts[3][2] - v_gap]
        
        new_pts = [n0, n1, n2, n3, n4, n5, n6, n7,
                n8, n9, n10, n11, n12, n13, n14, n15]
        x=[n[0] for n in new_pts]
        y=[n[1] for n in new_pts]
        z=[n[2] for n in new_pts]
        
        window_lines = go.Scatter3d(
            x=[x[0], x[1], x[2], x[3], x[0], None,
            x[4], x[5], x[6], x[7], x[4], None,
            x[8], x[9], x[10], x[11], x[8], None,
            x[12], x[13], x[14], x[15], x[12]],
            y=[y[0], y[1], y[2], y[3], y[0], None,
            y[4], y[5], y[6], y[7], y[4], None,
            y[8], y[9], y[10], y[11], y[8], None,
            y[12], y[13], y[14], y[15], y[12]],
            z=[z[0], z[1], z[2], z[3], z[0], None,
            z[4], z[5], z[6], z[7], z[4], None,
            z[8], z[9], z[10], z[11], z[8], None,
            z[12], z[13], z[14], z[15], z[12]],
            mode='lines',
            marker_color='black',
            marker_line_width=4
        )
        window_mesh = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[0, 0, 4, 4, 8, 8, 12, 12],
            j=[1, 3, 5, 7, 9, 11, 13, 15],
            k=[2, 2, 6, 6, 10, 10, 14, 14],
            showscale=False,
            flatshading=True,
            facecolor=['aquamarine', 'aquamarine', 'aquamarine', 'aquamarine',
                       'aquamarine', 'aquamarine', 'aquamarine', 'aquamarine']
        )
        
    if typology == 2:
        
        wall_area = w * h
        glaz_area = wall_area * wwr / 2.
        win_width = w * .95
        win_height = glaz_area / win_width
        h_gap = (w - win_width) / 2.
        v_gap = (h - win_height) / 2.
        
        # south // upper
        n0 = [pts[0][0] + h_gap, pts[0][1] - .1, pts[0][2] + v_gap + 5]
        n1 = [pts[1][0] - h_gap, pts[1][1] - .1, pts[1][2] + v_gap + 5]
        n2 = [pts[2][0] - h_gap, pts[2][1] - .1, pts[2][2] - v_gap + 5]
        n3 = [pts[3][0] + h_gap, pts[3][1] - .1, pts[3][2] - v_gap + 5]
        
        # north // upper
        n4 = [pts[4][0] + h_gap, pts[4][1] + .1, pts[4][2] + v_gap + 5]
        n5 = [pts[5][0] - h_gap, pts[5][1] + .1, pts[5][2] + v_gap + 5]
        n6 = [pts[6][0] - h_gap, pts[6][1] + .1, pts[6][2] - v_gap + 5]
        n7 = [pts[7][0] + h_gap, pts[7][1] + .1, pts[7][2] - v_gap + 5]
        
        # east // upper
        n8 = [pts[1][0] + .1, pts[1][1] + h_gap, pts[1][2] + v_gap + 5]
        n9 = [pts[5][0] + .1, pts[5][1] - h_gap, pts[5][2] + v_gap + 5]
        n10 = [pts[6][0] + .1, pts[6][1] - h_gap, pts[6][2] - v_gap + 5]
        n11 = [pts[2][0] + .1, pts[2][1] + h_gap, pts[2][2] - v_gap + 5]
        
        # west // upper
        n12 = [pts[0][0] - .1, pts[0][1] + h_gap, pts[0][2] + v_gap + 5]
        n13 = [pts[4][0] - .1, pts[4][1] - h_gap, pts[4][2] + v_gap + 5]
        n14 = [pts[7][0] - .1, pts[7][1] - h_gap, pts[7][2] - v_gap + 5]
        n15 = [pts[3][0] - .1, pts[3][1] + h_gap, pts[3][2] - v_gap + 5]
        
        # south // lower
        m0 = [pts[0][0] + h_gap, pts[0][1] - .1, pts[0][2] + v_gap - 5]
        m1 = [pts[1][0] - h_gap, pts[1][1] - .1, pts[1][2] + v_gap - 5]
        m2 = [pts[2][0] - h_gap, pts[2][1] - .1, pts[2][2] - v_gap - 5]
        m3 = [pts[3][0] + h_gap, pts[3][1] - .1, pts[3][2] - v_gap - 5]
        
        # north // lower
        m4 = [pts[4][0] + h_gap, pts[4][1] + .1, pts[4][2] + v_gap - 5]
        m5 = [pts[5][0] - h_gap, pts[5][1] + .1, pts[5][2] + v_gap - 5]
        m6 = [pts[6][0] - h_gap, pts[6][1] + .1, pts[6][2] - v_gap - 5]
        m7 = [pts[7][0] + h_gap, pts[7][1] + .1, pts[7][2] - v_gap - 5]
        
        # east // lower
        m8 = [pts[1][0] + .1, pts[1][1] + h_gap, pts[1][2] + v_gap - 5]
        m9 = [pts[5][0] + .1, pts[5][1] - h_gap, pts[5][2] + v_gap - 5]
        m10 = [pts[6][0] + .1, pts[6][1] - h_gap, pts[6][2] - v_gap - 5]
        m11 = [pts[2][0] + .1, pts[2][1] + h_gap, pts[2][2] - v_gap - 5]
        
        # west // lower
        m12 = [pts[0][0] - .1, pts[0][1] + h_gap, pts[0][2] + v_gap - 5]
        m13 = [pts[4][0] - .1, pts[4][1] - h_gap, pts[4][2] + v_gap - 5]
        m14 = [pts[7][0] - .1, pts[7][1] - h_gap, pts[7][2] - v_gap - 5]
        m15 = [pts[3][0] - .1, pts[3][1] + h_gap, pts[3][2] - v_gap - 5]
        
        new_pts = [n0, n1, n2, n3, n4, n5, n6, n7,
                n8, n9, n10, n11, n12, n13, n14, n15,
                m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, 
                m10, m11, m12, m13, m14, m15]
        
        x=[i[0] for i in new_pts]
        y=[i[1] for i in new_pts]
        z=[i[2] for i in new_pts]
        
        window_lines = go.Scatter3d(
            x=[x[0], x[1], x[2], x[3], x[0], None,
            x[4], x[5], x[6], x[7], x[4], None,
            x[8], x[9], x[10], x[11], x[8], None,
            x[12], x[13], x[14], x[15], x[12], None,
            x[16], x[17], x[18], x[19], x[16], None,
            x[20], x[21], x[22], x[23], x[20], None,
            x[24], x[25], x[26], x[27], x[24], None,
            x[28], x[29], x[30], x[31], x[28] 
            ],
            y=[y[0], y[1], y[2], y[3], y[0], None,
            y[4], y[5], y[6], y[7], y[4], None,
            y[8], y[9], y[10], y[11], y[8], None,
            y[12], y[13], y[14], y[15], y[12], None,
            y[16], y[17], y[18], y[19], y[16], None,
            y[20], y[21], y[22], y[23], y[20], None,
            y[24], y[25], y[26], y[27], y[24], None,
            y[28], y[29], y[30], y[31], y[28] 
            ],
            z=[z[0], z[1], z[2], z[3], z[0], None,
            z[4], z[5], z[6], z[7], z[4], None,
            z[8], z[9], z[10], z[11], z[8], None,
            z[12], z[13], z[14], z[15], z[12], None,
            z[16], z[17], z[18], z[19], z[16], None,
            z[20], z[21], z[22], z[23], z[20], None,
            z[24], z[25], z[26], z[27], z[24], None,
            z[28], z[29], z[30], z[31], z[28] 
            ],
            mode='lines',
            marker_color='black',
            marker_line_width=4
        )
        window_mesh = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28],
            j=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            k=[2, 2, 6, 6, 10, 10, 14, 14, 18, 18, 22, 22, 26, 26, 30, 30],
            showscale=False,
            flatshading=True,
            facecolor=['aquamarine', 'aquamarine', 'aquamarine', 'aquamarine',
                       'aquamarine', 'aquamarine', 'aquamarine', 'aquamarine',
                       'aquamarine', 'aquamarine', 'aquamarine', 'aquamarine',
                       'aquamarine', 'aquamarine', 'aquamarine', 'aquamarine']
        )
        
    if typology == 3:
        
        wall_area = w * h
        glaz_area = wall_area * wwr / 2.
        win_width = w * .95
        win_height = glaz_area / win_width
        h_gap = (w - win_width) / 2.
        v_gap = (h - win_height) / 2.
        
        # south
        n0 = [pts[0][0] + h_gap, pts[0][1] - .1, pts[0][2] + v_gap]
        n1 = [pts[1][0] - h_gap, pts[1][1] - .1, pts[1][2] + v_gap]
        n2 = [pts[2][0] - h_gap, pts[2][1] - .1, pts[2][2] - v_gap]
        n3 = [pts[3][0] + h_gap, pts[3][1] - .1, pts[3][2] - v_gap]
        
        m = midpoint(n0, n1)
        m0 = [m[0] - 0.25, m[1], m[2]]
        m1 = [m[0] + 0.25, m[1], m[2]]
        m = midpoint(n2, n3)
        m2 = [m[0] - 0.25, m[1], m[2]]
        m3 = [m[0] + 0.25, m[1], m[2]]
        
        # north
        n4 = [pts[4][0] + h_gap, pts[4][1] + .1, pts[4][2] + v_gap]
        n5 = [pts[5][0] - h_gap, pts[5][1] + .1, pts[5][2] + v_gap]
        n6 = [pts[6][0] - h_gap, pts[6][1] + .1, pts[6][2] - v_gap]
        n7 = [pts[7][0] + h_gap, pts[7][1] + .1, pts[7][2] - v_gap]
        
        m = midpoint(n4, n5)
        m4 = [m[0] - 0.25, m[1], m[2]]
        m5 = [m[0] + 0.25, m[1], m[2]]
        m = midpoint(n6, n7)
        m6 = [m[0] - 0.25, m[1], m[2]]
        m7 = [m[0] + 0.25, m[1], m[2]]
        
        # east
        n8 = [pts[1][0] + .1, pts[1][1] + h_gap, pts[1][2] + v_gap]
        n9 = [pts[5][0] + .1, pts[5][1] - h_gap, pts[5][2] + v_gap]
        n10 = [pts[6][0] + .1, pts[6][1] - h_gap, pts[6][2] - v_gap]
        n11 = [pts[2][0] + .1, pts[2][1] + h_gap, pts[2][2] - v_gap]
        
        # west
        n12 = [pts[0][0] - .1, pts[0][1] + h_gap, pts[0][2] + v_gap]
        n13 = [pts[4][0] - .1, pts[4][1] - h_gap, pts[4][2] + v_gap]
        n14 = [pts[7][0] - .1, pts[7][1] - h_gap, pts[7][2] - v_gap]
        n15 = [pts[3][0] - .1, pts[3][1] + h_gap, pts[3][2] - v_gap]
        
        all_pts = [n0, n1, n2, n3, n4, n5,
                   n6, n7, n8, n9, n10, n11,
                   n12, n13, n14, n15,
                   m0, m1, m2, m3, m4, m5, m6, m7]
        
        window_lines = go.Scatter3d(
            x=[n0[0], m0[0], m2[0], n3[0], n0[0], None,
               n1[0], m1[0], m3[0], n2[0], n1[0], None,
               n4[0], m4[0], m6[0], n7[0], n4[0], None,
               n5[0], m5[0], m7[0], n6[0], n5[0], None,
               n8[0], n9[0], n10[0], n11[0], n8[0], None,
               n12[0], n13[0], n14[0], n15[0], n12[0], None],
            y=[n0[1], m0[1], m2[1], n3[1], n0[1], None,
               n1[1], m1[1], m3[1], n2[1], n1[1], None,
               n4[1], m4[1], m6[1], n7[1], n4[1], None,
               n5[1], m5[1], m7[1], n6[1], n5[1], None,
               n8[1], n9[1], n10[1], n11[1], n8[1], None,
               n12[1], n13[1], n14[1], n15[1], n12[1], None],
            z=[n0[2], m0[2], m2[2], n3[2], n0[2], None,
               n1[2], m1[2], m3[2], n2[2], n1[2], None,
               n4[2], m4[2], m6[2], n7[2], n4[2], None,
               n5[2], m5[2], m7[2], n6[2], n5[2], None,
               n8[2], n9[2], n10[2], n11[2], n8[2], None,
               n12[2], n13[2], n14[2], n15[2], n12[2], None],
            mode='lines',
            marker_color='black',
            marker_line_width=4
        )
        x1 = [i[0] for i in all_pts]
        y1 = [i[1] for i in all_pts]
        z1 = [i[2] for i in all_pts]
        
        window_mesh = go.Mesh3d(
            x=x1,
            y=y1,
            z=z1,
            i=[0, 0, 1, 1, 8, 8, 5, 5, 4, 4, 12, 12],
            j=[16, 3, 17, 2, 9, 11, 21, 6, 20, 7, 13, 15],
            k=[18, 18, 19, 19, 10, 10, 23, 23, 22, 22, 14, 14],
            showscale=False,
            flatshading=True,
            facecolor=['aquamarine', 'aquamarine', 'aquamarine', 'aquamarine',
                       'aquamarine', 'aquamarine', 'aquamarine', 'aquamarine', 
                       'aquamarine', 'aquamarine', 'aquamarine', 'aquamarine']
        )
    return window_mesh, window_lines

def make_mesh(footprint, wwr, num_stories, num_units):
    
    if num_stories == 1 and num_units == 1:
        
        l = sqrt(footprint)
        w = sqrt(footprint)
        h = 10
        
        p0 = [0, 0, 0]
        p1 = [w, 0, 0]
        p2 = [w, 0, h]
        p3 = [0, 0, h]
        p4 = [0, w, 0]
        p5 = [w, w, 0]
        p6 = [w, w, h]
        p7 = [0, w, h]
        pts = [p0, p1, p2, p3, p4, p5, p6, p7]
    
        wall_mesh = go.Mesh3d(
            x=[0, 0, 0, 0, l, l, l, l],
            y=[0, w, w, 0, 0, w, w, 0],
            z=[0, 0, h, h, 0, 0, h, h],
            i=[0, 0, 7, 7, 4, 4, 1, 1],
            j=[1, 2, 4, 3, 5, 7, 5, 2],
            k=[2, 3, 0, 0, 6, 6, 6, 6],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke',
                       'whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke']
        )
        floor_mesh = go.Mesh3d(
            x=[0, 0, l ,l],
            y=[0, w, w, 0],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 3],
            k=[2, 2],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke']
        )
        ceiling_mesh = go.Mesh3d(
            x=[0, 0, l ,l],
            y=[0, w, w, 0],
            z=[h, h, h, h],
            i=[0, 0],
            j=[1, 3],
            k=[2, 2],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke']            
        ) 
        wall_lines = go.Scatter3d(
            x=[0, 0, 0, 0, 0, None, 
               l, l, l, l, l, None,
               0, l, None,
               0, l, None,
               0, l, None,
               0, l],
            y=[0, w, w, 0, 0, None, 
               0, w, w, 0, 0, None,
               0, 0, None,
               0, 0, None,
               w, w, None,
               w, w],
            z=[0, 0, h, h, 0, None, 
               0, 0, h, h, 0, None,
               0, 0, None,
               h, h, None,
               h, h, None,
               0, 0],
            mode='lines',
            marker_color='black',
            marker_line_width=4
        )
        window_mesh, window_lines = make_window(pts, w, h, wwr, 1)
        
    if num_stories == 2 and num_units == 1:
        
        l = sqrt(footprint)
        w = sqrt(footprint)
        h = 20
        
        p0 = [0, 0, 0]
        p1 = [w, 0, 0]
        p2 = [w, 0, h]
        p3 = [0, 0, h]
        p4 = [0, w, 0]
        p5 = [w, w, 0]
        p6 = [w, w, h]
        p7 = [0, w, h]
        pts = [p0, p1, p2, p3, p4, p5, p6, p7]
        
        wall_mesh = go.Mesh3d(
            x=[0, 0, 0, 0, l, l, l, l],
            y=[0, w, w, 0, 0, w, w, 0],
            z=[0, 0, h, h, 0, 0, h, h],
            i=[0, 0, 7, 7, 4, 4, 1, 1],
            j=[1, 2, 4, 3, 5, 7, 5, 2],
            k=[2, 3, 0, 0, 6, 6, 6, 6],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke',
                       'whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke']
        )
        floor_mesh = go.Mesh3d(
            x=[0, 0, l ,l],
            y=[0, w, w, 0],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 3],
            k=[2, 2],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke']
        )
        ceiling_mesh = go.Mesh3d(
            x=[0, 0, l ,l],
            y=[0, w, w, 0],
            z=[h, h, h, h],
            i=[0, 0],
            j=[1, 3],
            k=[2, 2],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke']            
        )
        wall_lines = go.Scatter3d(
            x=[0, 0, 0, 0, 0, None, 
               l, l, l, l, l, None,
               0, 0, l, l, 0, None,
               0, l, None,
               0, l, None,
               0, l, None,
               0, l],
            y=[0, w, w, 0, 0, None, 
               0, w, w, 0, 0, None,
               0, w, w, 0, 0, None,
               0, 0, None,
               0, 0, None,
               w, w, None,
               w, w],
            z=[0, 0, h, h, 0, None, 
               0, 0, h, h, 0, None,
               10, 10, 10, 10, 10, None,
               0, 0, None,
               h, h, None,
               h, h, None,
               0, 0],
            mode='lines',
            marker_color='black',
            marker_line_width=4
        )
        window_mesh, window_lines = make_window(pts, w, h, wwr, 2)
        
    if num_stories == 1 and num_units == 2:
        
        l = sqrt(footprint)
        w = sqrt(footprint)
        h = 10

        p0 = [0, 0, 0]
        p1 = [w, 0, 0]
        p2 = [w, 0, h]
        p3 = [0, 0, h]
        p4 = [0, w, 0]
        p5 = [w, w, 0]
        p6 = [w, w, h]
        p7 = [0, w, h]
        pts = [p0, p1, p2, p3, p4, p5, p6, p7]
               
        wall_mesh = go.Mesh3d(
            x=[0, 0, 0, 0, l, l, l, l],
            y=[0, w, w, 0, 0, w, w, 0],
            z=[0, 0, h, h, 0, 0, h, h],
            i=[0, 0, 7, 7, 4, 4, 1, 1],
            j=[1, 2, 4, 3, 5, 7, 5, 2],
            k=[2, 3, 0, 0, 6, 6, 6, 6],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke',
                       'whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke']
        )
        floor_mesh = go.Mesh3d(
            x=[0, 0, l ,l],
            y=[0, w, w, 0],
            z=[0, 0, 0, 0],
            i=[0, 0],
            j=[1, 3],
            k=[2, 2],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke']
        )
        ceiling_mesh = go.Mesh3d(
            x=[0, 0, l ,l],
            y=[0, w, w, 0],
            z=[h, h, h, h],
            i=[0, 0],
            j=[1, 3],
            k=[2, 2],
            showscale=False,
            flatshading=True,
            facecolor=['whitesmoke', 'whitesmoke']            
        )
        wall_lines = go.Scatter3d(
            x=[0, 0, 0, 0, 0, None, 
               l, l, l, l, l, None,
               l/2, l/2, l/2, l/2, l/2, None,
               0, l, None,
               0, l, None,
               0, l, None,
               0, l],
            y=[0, w, w, 0, 0, None, 
               0, w, w, 0, 0, None,
               0, w, w, 0, 0, None,
               0, 0, None,
               0, 0, None,
               w, w, None,
               w, w],
            z=[0, 0, h, h, 0, None, 
               0, 0, h, h, 0, None,
               0, 0, h, h, 0, None,
               0, 0, None,
               h, h, None,
               h, h, None,
               0, 0],
            mode='lines',
            marker_color='black',
            marker_line_width=4
        )
        window_mesh, window_lines = make_window(pts, w, h, wwr, 3)
    if wwr == 0:
        fig = go.Figure(data=[
            wall_mesh,
            floor_mesh,
            ceiling_mesh,
            wall_lines])
    else:
        fig = go.Figure(data=[
                            wall_mesh,
                            floor_mesh,
                            ceiling_mesh,
                            wall_lines,
                            window_mesh,
                            window_lines
                            ])

    fig.update_layout(showlegend=False,
                      hovermode=False,
                      scene_camera=dict(eye=dict(x=3, y=1.5, z=.8)),
                    #   camera_eye={'x':2, 'y':2, 'z':2},
                      margin={'pad':0,
                            'l':0,
                            'r':0,
                            'b':0,
                            't':0},
                    #   autosize=True
                    #   width=600,
                      height=375,
                    #   yaxis_fixedrange=True,
                    #   xaxis_fixedrange=True
                      )
    fig.update_scenes(
        # aspectratio={'x': 1.7, 'y': 1.7, 'z': 1},
                      xaxis_visible=False,
                      yaxis_visible=False,
                      zaxis_visible=False)
    # fig.show()
    # fig.write_image('filename.pdf', engine='orca')
    return fig
 
if __name__ == '__main__':
    fig = make_mesh(footprint=400, wwr=0.4, num_stories=1, num_units=2)
    fig.show()