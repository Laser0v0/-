# 无脑的数值积分算法
function nativeInt(f,a,b,n)
    h = (b-a)/n
    integ = 0
    for k = 1:n
        integ += f(a+k*h)*h
    end
    return integ
end

# 阶乘
function factorial(n)
    return  n<=1 ? 1 : n*factorial(n-1)
end


# 多项式乘法，形式为an*x^n，大端小端自定义，输出和输入的格式相同
# 建议a[1]为最低位常数项
function polyMulti(a,b)
    na = length(a); nb = length(b)
    c = zeros(na+nb-1)
    for i = 1:na
        c[i:i+nb-1] += a[i]*b
    end
    return c
end

# 多项式除法，a[1]为最低位 a/b = c ... a
function polyDiv(a,b)
    a = copy(a)         #此为余数
    na = length(a); nb = length(b)
    nc = na-nb+1
    c = zeros(nc)
    for i = 0:nc-1
        c[nc-i] = a[end-i] ÷ b[end]
        a[nc-i:end-i] += -c[nc-i]*b[1:end]
    end
    return c,a
end

#将(x-0)(x-1)...(x-n)化成an*x^n形式,A[1]为最低位(常数项)
#输入为n输出为[a1,a2,...,an]
function polyFromNum(n)
    return n==0 ? [0,1] : polyMulti(
        [-n,1],polyFromNum(n-1))
end

#将(x-x1)(x-x2)...(x-xn)化成an*x^n形式,A[1]为最低位(常数项)
#输入为[x1,x2,...,xn]输出为[a1,a2,...,an]
function polySimplify(x)
    return length(x)==1 ? [-x[1],1] : polyMulti(
        [-x[1],1],polySimplify(x[2:end]))
end

#多项式积分，输入为([a0,a1,...,an],aStart,aEnd)
#输出为[b0,b1,...,bn+1]
function polyInf(a,as,ae)
    #因为积分后位数升1，所以幂数从1开始
    return sum([(ae^i-as^i)*a[i]/i for i in 1:length(a)])    
end

# 获取Cotes系数
function getCotes(n,k=[])
    if k == []
        return [getCotes(n,i) for i in 0:n]
    end
    kerr = polySimplify([i for i in 0:n if i != k])
    cotes = (-1)^(n-k)/n/factorial(k)/factorial(n-k)
    return cotes*polyInf(kerr,0,n)
end

# Newton-Cotes公式
function NewtonCotes(f,a,b,n)
    cotes = getCotes(n)
    return sum([cotes[i]*f(a+(b-a)/n*(i-1)) for i in 1:n+1])
end

# 复化梯形公式,f为函数，a,b为积分区间，err为要求精度
# flag为1，2，3，4分别表示梯形、Simpson、Cotes、Romberg算法
function Trapz(f,a,b,flag=1,err=0.01,nMax=10)
    flag = min(flag,nMax)       #
    nMax = 2^nMax

    Tn = zeros(flag)                        #存储梯形公式的值
    n = 1
    h = b-a
    T = h/2*(f(a)+f(b))
    Tn[end] = T
    myErr = err + 1                         #初始误差大于目标误差

    while myErr > err && n < nMax
        newX = [2*i-1 for i in 1:n]         #新的x序列
        n = n*2; h = h/2                    #更新n，h值
        new  = sum(f.(a.+h*newX))*h
        myErr = abs(T/2-new)
        T = T/2 + new                       #新的梯形值
        if flag > 1
            Tn[1:end-1] = Tn[2:end]
        end
        Tn[end] = T                         #更新梯形公式的值
    end

    if flag > 1
        Sn = (4*Tn[2:end]-Tn[1:end-1])/3
        Tn = [Tn[end],Sn[end]]
    end

    if flag > 2
        Cn = (16*Sn[2:end]-Sn[1:end-1])/15      #Cotes公式
        Tn = vcat(Tn,Cn[end])
    end

    if flag > 3
        Rn = (64*Cn[2:end]-Cn[1:end-1])/63      #Romberg公式
        Tn = vcat(Tn,Rn[end])
    end

    return Tn,myErr
end