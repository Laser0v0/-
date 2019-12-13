# Gauss消去法
# M为n×n+1增广矩阵，对应Ax=b,M[:,end]为b
function GaussElimination(M)
    M = copy(M)
    n,_ = size(M)
    # 消元过程
    for i in 1:n-1
        vMinus = M[i,:]/M[i,i]  #当前行循环所要消去的变量
        for j in i+1:n
            M[j,:] -= vMinus*M[j,i]
        end
    end
    # 回代过程
    x = zeros(n)
    for i in n:-1:1
        sumValue = sum([M[i,j]*x[j] for j in i:n])
        x[i] = (M[i,end]-sumValue)/M[i,i]
    end
    return x
end

# 生成单位矩阵
function eye(n)
    M = zeros(n,n)
    for i in 1:n
        M[i,i] = 1
    end
    return M
end


# 矩阵的LU分解
function LU(M)
    U = copy(M)
    n,_ = size(M)
    L = eye(n)
    eleMatrix = eye(n)
    # 初等变换过程
    for i in 1:n-1
        eleMatrix[i+1:end,i] = [-U[j,i]/U[i,i] for j in i+1:n]
        U = eleMatrix*U
        L = L/eleMatrix
        eleMatrix[i+1:end,i] = [0 for j in i+1:n]
    end
    return L,U
end

# 生成完全主元素矩阵,注意保护变量
# M为n×n+1增广矩阵，对应Ax=b,M[:,end]为b
function allPrinciple(M)
    M = copy(M)
    n,_ = size(M)
    J = Int.(zeros(n-1))#保存x的交换信息
    for i = 1:n-1
        pos = findmax(M[i:end,i:end-1])[2]
        ni = pos[1]+i-1; nj = pos[2]+i-1    #最大值所在位置
        J[i] = nj
        exchange(M,i,ni,"row")  #交换行
        exchange(M,i,nj,"col")  #交换列
    end
    return M,J
end

#全主元素消去法
function allElimitation(M)
    M,J = allPrinciple(M)       #最大主元替换
    x = GaussElimination(M)     #高斯消去法
    for i in length(J):-1:1     #未知数换位
        exchange(x,i,J[i],"arr")
    end
    return x
end

# Gauss-Jordan消去法
# M为n×n+1增广矩阵，对应Ax=b,M[:,end]为b
function GaussJordan(M,maxFlag=true)
    if maxFlag
        M,J = allPrinciple(M)       #最大主元替换
    end

    n,_ = size(M)
    # Gauss消元
    for i in 1:n-1
        vMinus = M[i,:]/M[i,i]  #当前行循环所要消去的变量
        for j in i+1:n
            M[i+1:n,:] -= [vMinus*M[j,i] for j in i+1:n]
        end
    end

    # Jordan消元
    for i in n:-1:2
        vMinus = M[i,:]/M[i,i]
        for j in 1:i-1
            M[j,:] -= vMinus*M[j,i]
        end
    end

    # 回代
    x = [M[i,end]/M[i,i] for i in 1:n]
    if maxFlag
        for i in length(J):-1:1     #未知数换位
            exchange(x,i,J[i],"arr")
        end
    end
    return M,x

end


# 交换行或者列的位置
function exchange(M,i,j,flag="row")
    if flag == "row"
        temp = M[i,:]
        M[i,:] = M[j,:]
        M[j,:] = temp
    elseif flag == "col"
        temp = M[:,i]
        M[:,i] = M[:,j]
        M[:,j] = temp
    elseif flag == "arr"
        temp = M[i]
        M[i] = M[j]
        M[j] = temp
    end
end

# 追赶法,输入M为三对角矩阵，f为向量
function chasing(M,f)
    n,_ = size(M)
    b = [M[i,i] for i in 1:n]
    c = [M[i,i+1] for i in 1:n-1]
    a = vcat(1,[M[i+1,i] for i in 1:n-1])

    # 根据递推公式求beta
    beta = zeros(n-1)
    beta[1] = c[1]/b[1]
    for i in 2:n-1
        beta[i] = c[i]/(b[i]-a[i]*beta[i-1])
    end

    # 求解y
    y = zeros(n)
    y[1] = f[1]/b[1]
    for i in 2:n
        y[i] = (f[i]-a[i]*y[i-1])/(b[i]-a[i]*beta[i-1])
    end

    # x的追赶过程
    x = zeros(n)
    x[n] = y[n]
    for i in n-1:-1:1
        x[i] = y[i]-beta[i]*x[i+1]
    end
    
    return x
end

## Jocobi迭代法
## n为迭代次数
function Jacobi(M,b,n=10)
    nDim,_ = size(M)
    D = zeros(nDim,nDim)
    for i in 1:nDim
        D[i,i] = M[i,i]
    end
    B = D\(D-M)
    f = D\b

    x = f       #初始化
    for i = 1:n
        x = B*x+f
    end
    return x
end

# Gauss-Seidel法
function GaussSeidel(M,b,n=10)
    
end

# 乘幂法
function Power(M,n)
    arr = rand(size(M)[1])
    for i in 1:n-1
        arr = M*arr
        arr = arr/maximum(arr)  #对向量进行归一化
    end
    Arr = M*arr                 
    return Arr,Arr./arr
end

# 加速之后的幂法
function acPower(M,n,p)
    B = M .- eye(size(M)[1])*p
    arr,val = Power(B,n)
    return arr,val.+p
end

# Schmidt正交化法，输入为矩阵
function Schmidt(M)
    n,_ = size(M)
    N = zeros(n,n)
    for i = 1:n
        B = M[:,i]
        for j in 1:i-1
            B -= M[:,i]'*N[:,j]*N[:,j]
        end
        N[:,i] = B/sqrt(sum([B[j]^2 for j in 1:n]))
    end
    return N
end

# Jacobi旋转法
function JacobiRot(M)
    n,_=size(M)
    E = eye(n)
    for i = 1:n
        for j = 1:n
            p2Tan = -2*M[i,j]/M[i,i]-M[j,j]
            p2Cos = 1/sqrt(1+p2Tan^2)
            p2Sin = p2Tan*p2Cos
            pCos = sqrt(p2Cos+1)/2
            pSin = p2Sin/pCos/2
            
            cii = M[i,i]*pCos^2+M[j,j]*pSin^2-M[i,j]*p2Sin
            cjj = cii+2*M[i,j]*p2Sin
            cij = (M[i,i]-M[j,j])*p2Sin/2+M[i,j]*p2Cos
            ci = [M[i,k]*pCos-M[j,k]*pSin for k in 1:n]
            ci[i] = cii;ci[j]=cij
            cj = [M[j,k]*pCos+M[i,k]*pSin for k in 1:n]
            cj[i] = cij;cj[j]=cjj

            M[i,:] = ci;M[:,i] = ci
            M[j,:] = cj;M[:,j] = cj
        end
    end
    return M
end