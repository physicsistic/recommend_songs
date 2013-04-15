from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time
import random as rand

TRACKS = 10000
BINS = 25
def pre_processing():
  data = open("song10000", "r").readlines()
  isomap_data = open("song_reduced_10000", "r").readlines()

  matrix = []

  print("Importing data from file...")
  current_time = time.clock()
  for line in data:
	sline = line.split('\t')
	sline[2] = sline[2].split(',')[0] # easy genre assignment
	matrix.append(sline)

  print "Time to import files: %6.2f s" % (time.clock() - current_time)

  M = np.array(matrix)[:,0:4]

  matrix = []
  for line in isomap_data:
    matrix.append(str.split(line))

  I = np.array(matrix, dtype=np.float32)
  I.reshape((TRACKS,1))
  M = np.hstack((M, I))

  # histogram of raw isomap data
  n, bins, patches = plt.hist(I, BINS)
  plt.title('Raw Isomap Data')
  plt.xlabel('Score')
  plt.ylabel('Frequency')
  plt.show()

  return M, bins

def find_bin(bins, score):
  i = 0
  while i < len(bins) and score > bins[i]:
    i += 1
  return i

def sort(d, bins, g, s, track, artist):
  # d = {genre: {bin: [track list]}}
  if g in d:
    b = find_bin(bins,s)
    if b in d[g]:
      d[g][b][0] += [track + ', ' + artist]
      d[g][b][1] += 1
    else:
      d[g][b] = [[track +', '+ artist], 1]
  else:
    d[genre] = {find_bin(bins,s): [[track + ', ' + artist], 1]}
  
  return d

def genre_reduce(x, y):
  # y = {genre: [score (list)]}
  for genre, scores  in y.iteritems():
    if genre in x:
	  x[genre] += scores
    else:
      x[genre] = scores
  return x

def bin_reduce(x, y):
  # y = {genre: {bin: [track list]}}
  for genre, bins in y.iteritems():
    if genre in x:
      for b, track_list in bins.iteritems():
        if b in x[genre]:
          x[genre][b] += track_list
        else:
          x[genre][b] = track_list
    else:
      x[genre] = bins

  return x

#print("Writing data to file...")
#current_time = time.clock()
#results = open("processed_1000", "w")
#for line in M:
#	results.write("%s" % line)
#	results.write("\n")
#results.close()
#print "Time to write files: %6.2f s" % (time.clock() - current_time)

if __name__ == '__main__':
  # Intialize MPI constants
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  fields = 5

  # Read the data from file
  if rank == 0:
    data, bins = pre_processing()
  local_result = {}

  comm.barrier()
  p_start = MPI.Wtime()

  # Distribute with scatter
  if rank == 0:
    data = data.reshape((size, TRACKS*fields/size))
#    local_data = np.empty((1,TRACKS*fields/size))
  else:
    data = None
 #   local_data = np.empty((1,TRACKS*fields/size))

  #comm.Scatter(sendbuf=data, recvbuf=local_data, root=0)
  local_data = comm.scatter(data, root=0)
  local_data = local_data.reshape((TRACKS/size, fields))
  # Create track list
  for k in xrange(0, TRACKS/size):
    # Collect genres
    row = local_data[k,:]
    genre = row[2]
    score = float(row[fields-1])
    if genre in local_result:
      local_result[genre] += [score]
    else:
      local_result[genre] = [score]

  result = comm.reduce(local_result, op=genre_reduce, root=0)

  comm.barrier
  p_stop = MPI.Wtime()

  if rank==0:
    print p_stop-p_start
    # plot results
    x, y, y1 = [], [], [0]
    resultreverse = {}
    # General Scatter - score vs. hashed genre
    for k,v in result.iteritems():
      #for score in v:
       # x.append(hash(k)/(1e19))
        #y.append(score)
      if len(v) in resultreverse:
        resultreverse[len(v)] += [k]
      else:
        resultreverse[len(v)] = [k]

    topten = sorted(resultreverse.items(), reverse=True)[:15]
    toptengenres = []
    for g in topten:
      toptengenres.extend(g[1])
    histograms, l = 0, len(toptengenres)
    for k,v in result.iteritems():
      if k in toptengenres:
        hist = np.zeros(BINS+1) # number of bins
        for score in v:
          x.append(toptengenres.index(k)/float(l))
          y.append(score)
          # count into appropriate bin
          i = find_bin(bins, score)
          hist[i] +=1
        c = toptengenres.index(k)/float(l)
        p = plt.bar(bins, hist, width=bins[2]-bins[1], color=(c,c/2.,rand.random()), bottom=histograms) 
        histograms += hist
        # plt.legend(p[0], k)

    plt.show()
 
    plt.scatter(x, y, c=x)
    plt.title('Most Popular Genres')
    plt.xlabel('Genre')
    plt.ylabel('Score')
    plt.xticks(np.arange(l)/float(l), toptengenres, rotation=60)
    plt.show() 
  else:
    toptengenres = None
    bins = None

  # Put tracks into genres + bins
  comm.barrier()
  p_start = MPI.Wtime()
  top = comm.bcast(toptengenres) 
  bins = comm.bcast(bins)

  top_tracks = {}
  other_tracks = {}
  # Create track list
  for k in xrange(0, TRACKS/size):
    # Collect genres
    row = local_data[k,:]
    genre = row[2]
    score = float(row[fields-1])
    if genre in top:
      top_tracks = sort(top_tracks, bins, genre, score, row[1], row[3])
    else:
      other_tracks = sort(other_tracks, bins, genre, score, row[1], row[3])

  top_results = comm.reduce(top_tracks, op=bin_reduce, root=0)
  all_results = comm.reduce(dict(top_tracks.items() + other_tracks.items()), op=bin_reduce, root=0)

  comm.barrier
  p_stop = MPI.Wtime()

  if rank==0:
    print p_stop-p_start
    y2, x2, annot, size = [], [], [], []
    l = len(toptengenres)
    for k,v in top_results.iteritems():
      for bin in v:
        y2.append(toptengenres.index(k)/float(l))
        x2.append(bin)
        annot.append(v[bin][0])
        size.append(25*v[bin][1])

    fig = plt.figure()

    def onpick(event):
      ind = event.ind
      xdata = event.mouseevent.xdata
      ydata = event.mouseevent.ydata
      a = np.take(annot, ind)
      annotation = ''
      for track in a[0]:
        annotation += track + '\n'
      plt.annotate(annotation, xy=(xdata, ydata),  xycoords='data',
                xytext=(-40, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="->",
                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )

      fig.canvas.draw()

    ax = fig.add_subplot(111)
    ax.scatter(x2, y2, c=y2, s=size, picker=True)
    plt.yticks(np.arange(l)/float(l), toptengenres, rotation=20)
    fig.canvas.mpl_connect('pick_event', onpick)
    ax.set_title('Most Popular Genres')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Genre')
    plt.axis([-1, BINS+1, -.1, 1.1])
    plt.show()

    y2, x2, annot, size = [], [], [], []
    genres = all_results.keys()
    l = len(genres)
    for k,v in all_results.iteritems():
      for bin in v:
        y2.append(genres.index(k)/float(l))
        x2.append(bin)
        annot.append(v[bin][0])
        size.append(25*v[bin][1])

    plt.scatter(x2, y2, c=y2, s=size, picker=True)
    # plt.yticks(np.arange(l)/float(l), genres)
    #fig.canvas.mpl_connect('pick_event', onpick)
    plt.title('All Genres (top 10)')
    plt.xlabel('Bin')
    plt.ylabel('Genre')
    plt.show()
 
