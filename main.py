import matplotlib.pyplot as plt
import webbpsf
import webbpsf.constants as const

corners = const.JWST_PRIMARY_SEGMENTS

A1 = corners[0][1].T


fig, ax = plt.subplots()
for segment in corners:
    corner = segment[1].T
    ax.scatter(corner[0], corner[1])
# ax.scatter(A1[0], A1[1])
ax.set(aspect='equal',
       title='Primary Segments Corners',
       xlabel=r'$x$ [m]',
       ylabel=r'$y$ [m]',
       )
plt.show()
