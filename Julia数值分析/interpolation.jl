using LinearAlgebra

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

#将(x-x1)(x-x2)...(x-xn)化成an*x^n形式,A[1]为最低位(常数项)
#输入为[x1,x2,...,xn]输出为[a1,a2,...,an]
function polySimplify(x)
    if length(x) == 1
        return [-x[1],1]
    else
        return polyMulti([-x[1],1],polySimplify(x[2:end]))
    end
end


# 连乘函数，输入为数组，输出为数组中所有数的乘积
function sumMulti(x)
    if length(x) == 1
        return x
    else
        return x[1]*sumMulti(x[2:end])
    end
end

# Language插值函数
# x y: 维度相同的列变量
# A: 多项式系数，A[1]为最低位
function Lagrange(x,y)
    n = length(x)
    xMat = x .- x' + Matrix{Float64}(I,n,n)
    polyMat = polySimplify(x)
    A = zeros(n)
    for i = 1:n
        quot = polyDiv(polyMat,[-x[i],1])[1]
        A += y[i] * quot ./ sumMulti(xMat[i,:])
    end 
    return A
end


# 差商算法,x,y为同等长度的数组
function diffQuotient(x,y)
    if length(x) == 2
        return (y[2]-y[1])/(x[2]-x[1])
    elseif length(x) == 1
        return y[1]
    else
        return (diffQuotient(x[2:end],y[2:end])
               -diffQuotient(x[1:end-1],y[1:end-1]))/(x[end]-x[1])
    end
end

# Newton插值函数
function Newton(x,y)
    n = length(x)
    A = zeros(n)
    hisQuot = y[1]      #保存差商的值
    hisPoly = [1]
    A[1] = y[1]
    for i = 2:n
        hisQuot = diffQuotient([x[1],x[i]],[hisQuot,diffQuotient(x[2:i],y[2:i])])
        hisPoly = polyMulti(hisPoly,[-x[i-1],1])
        A[1:length(hisPoly)] += hisQuot*hisPoly
    end
    return A
end

# Hermite插值函数
# x,y分别为自变量、因变量，m为导数
function Hermite(x,y,m)
    n = length(x)
    unitMat = Matrix{Float64}(I,n,n)    #单位矩阵
    xMat = x .- x' + unitMat
    xInverse = 1 ./xMat - unitMat
    polyMat = polySimplify(x)
    A = zeros(n*2)
    for i = 1:n
        #Lagrange基函数
        lag = polyDiv(polyMat,[-x[i],1])[1] ./ sumMulti(xMat[i,:])  
        lag = polyMulti(lag,lag)
        sumInverse = 2*sum(xInverse[i,:])*y[i]
        a0 = y[i]+x[i]*(sumInverse-m[i])
        a1 = -sumInverse+m[i]
        A += polyMulti(lag,[a0,a1])
    end
    return A
end

# 得到三弯矩法中的弯矩值
# flag表示三类边界条件，需要注意，二类为周期条件，三类为
# para为三类边界条件输入的边界参数
function getMoment(x,y,flag,para=[0,0])
    n = length(x)
    h = x[2:end]-x[1:end-1]
    λ = h[2:end] ./ (h[1:end-1]+h[1:end-1])
    μ = 1 .- λ
    d = [6 * diffQuotient(x[i:i+2],y[i:i+2]) 
            for i in 1:n-2]
    
    nMatrix = Int(n-(-1.5*flag^2+6.5*flag-5))
    M = Matrix{Float64}(I,nMatrix,nMatrix)*2

    # 确定d,λ,μ的值
    if flag == 1
        d0 = 6/h[1]*((y[2]-y[1])/h[1]-para[1])
        dn = 6/h[end]*(-(y[end]-y[end-1])/h[end]+para[2])
        d = vcat(d0,d,dn)
        λ = vcat(1,λ)
        μ = vcat(μ,1)
    elseif flag == 2
        d[1] -= μ[1]*para[1]
        d[end] -= λ[end]*para[2]
        μ = μ[2:end]
    else
        dn = 6/(h[1]+h[end])*((y[2]-y[1])/h[1]-(y[end]-y[end-1])/h[end])
        d = vcat(d,dn)
        M[1,end] = μ[1]
        M[end,1] = λ[end]
        λ = vcat(λ,h[1]/(h[1]+h[end]))
        μ = vcat(μ[2:end],h[end]/(h[1]+h[end]))

    end

    # 求M矩阵
    for i in 1:nMatrix-1
        M[i,i+1] = λ[i]
        M[i+1,i] = μ[i]
    end

    return M,d
end



# 三弯矩法，x，y为自变量、因变量，m为二阶导
function moment3(x,y,m)
    n = length(x)
    A = zeros(n-1,4)
    h = [x[i+1]-x[i] for i = 1:n-1]
    c1 = [y[i]-m[i]*h[i]^2/6 for i = 1:n-1]
    c2 = [y[i+1]-m[i+1]*h[i]^2/6 for i = 1:n-1]
    x2 = x.^2
    x3 = x2.*x

    for i = 1:n-1
        p3 = (m[i]-m[i+1])/h[i]/6
        p2 = 3*(m[i+1]*x[i]-m[i]*x[i+1])/h[i]/6
        p1 = ((m[i]*x2[i+1]-m[i+1]*x2[i])/6-(c1[i]+c2[i]))/h[i]
        p0 = ((-m[i]*x3[i+1]+m[i+1]*x3[i])/6-(c1[i]*m[i+1]+c2[i]*m[i]))/h[i]
        A[i,:] = [p0,p1,p2,p3]
    end
    return A
end
