# 二分法求根，f为函数，err为误差
# 要求f(a)*f(b)<0
function Dichotomy(f,a,b,err)
    c = (a+b)/2
    val = f(c)
    #如果f(c)和f(a)异号，则将c赋值给b，否则将c赋值给a
    val*f(a)<0 ? b=c : a=c
    #如果大于误差，则继续二分，否则输出c
    return abs(val)>err ? Dichotomy(f,a,b,err) : c
end

# Newton法
# f为函数，x为初始值，err为误差
function Newton(f,x,err)
    fx = f(x)
    df = (fx-f(x-err))/err      #近似斜率
    xNew = x-fx/df
    # 如果满足精度要求则返回，否则继续迭代
    return abs(xNew-x) > err ? Newton(f,xNew,err) : xNew
end

# Euler法
# f为dy/dx=f(x,y), x为输入区间[x1,x2]，n为步长
# nIter 为梯形格式的迭代次数
function Euler(f,x,y0,n,nIter=0)
    h = (x[2]-x[1])/n
    x = [x[1]+i*h for i in 1:n]
    y = zeros(n)
    y[1] = y0
    for i in 2:n
        y[i] = y[i-1]+h*f(x[i],y[i-1])
        for j in 1:nIter
            y[i] = y[i-1]+h*f(x[i],y[i])
        end
    end
    return y
end

# Runge-Kutta法
# f为dy/dx=f(x,y), x为输入区间[x1,x2]，n为步长
# deg为阶数,支持2，3，4阶
# 二阶需要输入参数p,lmd1,lmd2;三阶需要额外输入q,r,s,lmd3
# 四阶需要另输入a,b,c,d,lmd4
function RungeKutta(f,x,y0,n;deg=2,kwargs...)
    h=(x[2]-x[1])/n
    x = [x[1]+i*h for i in 1:n]
    y = zeros(n)
    y[1] = y0

    
    # 初始化参数
    if deg == 2
        p,lmd1,lmd2 = isempty(kwargs) ? [1/2,1/2,1/2] : [
            kwargs[:p],kwargs[:lmd1],kwargs[:lmd2]]
    elseif deg == 3
        p,q,r,s,lmd1,lmd2,lmd3 = isempty(kwargs) ? [
            1/2,1,-1,2,1/6,2/3,1/6] : [kwargs[:p],
            kwargs[:q],kwargs[:r],kwargs[:s],
            kwargs[:lmd1],kwargs[:lmd2],kwargs[:lmd3]]
    elseif deg == 4
        p,q,r,s,a,b,c,d,lmd1,lmd2,lmd3 = isempty(kwargs) ? [
            1/2,1/2,0,0,1,0,1,0,1/6,1/3,1/3,1/6] : [
            kwargs[:p],kwargs[:q],kwargs[:r],kwargs[:s],
            kwargs[:a],kwargs[:b],kwargs[:c],kwargs[:d],
            kwargs[:lmd1],kwargs[:lmd2],kwargs[:lmd3],kwargs[:lmd4]]
    end


    for i in 2:n
        K1 = f(x[i-1],y[i-1])
        K = K1
        if deg>1
            K2 = f(x[i-1]+p*h,y[i-1]+p*h*K1)        #参数p
            K = lmd1*K1+lmd2*K2
        end
        if  deg>2
            K3 = f(x[i-1]+q*h,y[i-1]+q*h*(r*K1+s*K2))#参数q,r,s
            K += lmd3*K3                   #参数lmd3
        end
        if deg>3
            K4 = f(x[i-1]+a*h,y[i-1]+a*h*(b*K1+c*K2+d*K3))
            K += lmd4*k4
        end
        y[i] = y[i-1]+h*K
    end
    return y
end

