from FockSystem.FockSystem import OperSequence
c_down = OperSequence(0)
c_up = OperSequence(2)
a_up = ~c_up
a_down = ~c_down
def kitaev_chain(N):
   
    ECT,MU,CAR = OperSequence(),OperSequence(),OperSequence()
    ## Add ECT terms
    for i in range(N-1):
        ECT += ((c_down*(a_down>>1))>>i)
    ## Add CAR terms
    for i in range(N-1):
        CAR+= ((((a_down>>1))*a_down)>>i)
    ## Add mu terms
    for i in range(N):
        MU += ((c_down*(a_down))>>i)
    return MU,ECT,CAR


def kitaev_chain_spinful(N):
    ECT, MU, CAR,U = OperSequence(), OperSequence(), OperSequence(), OperSequence()

    ## Add ECT terms
    for i in range(N - 1):
        ECT += ((c_down * (a_down >> 1)) >> i) + ((c_up * (a_up >> 1)) >> i)
    ## Add CAR terms
    for i in range(N - 1):
        CAR += (((a_down >> 1) * a_up) >> i) - (((a_up >> 1) * a_down) >> i) 
    ## Add mu terms
    for i in range(N):
        MU += (c_down * (a_down)) >> i
        MU += (c_up * (a_up))>>i

    for i in range(N):
        U += (c_up*c_down*a_up*a_down) >> i
        
    return MU,U, ECT, CAR

def kramers_chain(N):
    ECT,MU, CAR =  OperSequence(), OperSequence(), OperSequence()
      ## Add ECT terms
    for i in range(N - 1):
        ECT += ((c_down * (a_down >> 1)) >> i) + ((c_up * (a_up >> 1)) >> i)
    ## Add CAR terms
    for i in range(N - 1):
        CAR += (((a_down >> 1) * a_up) >> i) - (((a_up >> 1) * a_down) >> i) 
    ## Add mu terms
    for i in range(N):
        MU += (c_down * (a_down)) >> i
        MU += (c_up *(a_up)) >> i
    return MU,ECT,CAR