from FockSystem.FockSystem import OperSequence

def kitaev_chain(N):
    c_down = OperSequence(0)
    c_up = OperSequence(2)
    a_up = ~c_up
    a_down = ~c_down
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