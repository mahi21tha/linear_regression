import numpy as np
def gradient_desent(x,y):
  learning_rate=0.01
  n=len(x)
  m_curr=b_curr=0
  iteration=20
  for i in range(iteration):
    y_pre=m_curr*x+b_curr
    cost=(1/n)*sum((y-y_pre)**2)
    md=-(2/n)*sum(x*(y-y_pre))
    bd=-(2/n)*sum((y-y_pre))
    m_curr=m_curr-learning_rate*md
    b_curr=b_curr-learning_rate*bd
    print("m{},b{},iteration{},cost{}".format(m_curr,b_curr,i,cost))




x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
gradient_desent(x,y)
