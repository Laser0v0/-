# 插值法

设函数$y=f(x)$在区间$[a,b]$上有定义，且在点$a\leqslant x_0\leqslant x_1\ldots\leqslant x_n\leqslant b$上的值$y_0, y_1 \ldots y_n$之间存在一个函数$P(x)$,使得

$$P(x_i)=y_i (i=0,1,\ldots, n)$$

成立，则称$P(x)$为$f(x)$的插值函数，在$x_0, x_1,\ldots,x_n$称为插值节点，包含节点的区间称为插值区间。如果$P(x)$是次数不超过$n$的代数多项式，则称之为插值多项式。

## Lagrange插值

### 数学原理

若我们有$n+1$组一一对应的$(x,y)$点对，假设存在一多项式$L_n(x)$，使之满足$L_n(x_j)=y_j,j=0,1,\ldots,n$，即

$$\left\{\begin{aligned}
a_0+a_1x_0+\ldots+a_nx_0^n&=&y_0\\
a_0+a_1x_1+\ldots+a_nx_1^n&=&y_1\\
&\vdots&\\
a_0+a_1x_n+\ldots+a_nx_n^n&=&y_n\\
\end{aligned}\right.
$$

则$L_n$即为该组数据的插值多项式。对于任意$j=0,1,\ldots,n$，有$L_n(x_j)=y_j$,则可令

$$
L_n(x)=\displaystyle\sum_{k=0}y_kl_k(x)
$$

其中，

$$
l_k(x_j)=\delta_{jk}=\left\{\begin{aligned}
1, k=j\\
0, k\not=j
\end{aligned}\right.
(j,k=0,1,\ldots,n)
$$

则可满足$L_n(x_j)=\displaystyle\sum_{k=0}^ny_kl_k(x_j)=y_j$，其中$l_k$被称为基函数。根据其所满足的要求，易得基函数的表达式为

$$
l_k(x)=\frac{(x-x_0)\ldots(x-x_{k-1})(x-x_{k+1})\ldots(x-x_n)}{(x_k-x_0)\ldots(x_k-x_{k-1})(x_k-x_{k+1})\ldots(x_k-x_n)}
$$

自此，我们就得到了Lagrange插值多项式。该式简洁明了，其插值基函数规律明显，能使人过目不忘。若记

$$\omega_{n+1}(x)=(x-x_0)(x-x_1)\ldots(x-x_n)$$

则

$$\omega'_{n+1}(x_k)=(x_k-x_0)\ldots(x_k-x_{k-1})(x_k-x_{k+1})\ldots(x_k-x_n)$$

那么Lagrange插值公式可写为

$$
L_n(x)=\displaystyle\sum_{k=0}^ny_k\frac{\omega_{n+1}(x)}{(x-x_k)\omega'_{n+1}(x_k)}
$$

特别地，对于两点而言，Lagrange插值退化为线性插值，即

$$
L_2(x)=y_1\frac{x-x_2}{x_1-x_2}+y_2\frac{x-x_1}{x_2-x_1}
$$

### 算法化

在Lagrange插值公式中，$\omega_{n+1}(x)$为多项式乘法。Julia中虽然对函数式编程有着良好的支持，但如果将函数本身作为输出，则函数实现细节将被隐藏，因此我们并不能得到真正的拟合参数。所以我们需要对多项式进行抽象，即将其简化为数组的形式，其表现形式为`[a_0,a_1,...,a_n]`,代表$a_0+a_1x+\ldots+a_nx^n$。

那么多项式的乘法可以通过进位的方式实现。对于常数项而言，并不会改变多项式的幂数；对于一阶项来说，乘以多项式数组，则将该数组向高位移一位。其实现方法为

```julia
# 多项式乘法，形式为an*x^n，建议a[1]为最低位常数项
function polyMulti(a,b)
    na = length(a); nb = length(b)
    c = zeros(na+nb-1)
    for i = 1:na
        c[i:i+nb-1] += a[i]*b
    end
    return c
end
```

在Lagrange插值公式中亦有多项式除法，其实现方法与乘法类似，只不过其运算顺序需要从高位开始，并且被除多项式的阶数要不小于除数多项式。
在此需要注意，和许多编程语言类似，Julia中的对象采用地址引用的形式，在传参之后如果直接改变数组内容，将穿透作用域，使得函数之外的数组也发生变化，因此需要对其进行深拷贝。

```julia
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
```

此外，考虑到Lagrange插值公式中包含连乘运算。其中，分子中的连乘为多项式连乘，分母中的多项式为值连乘，所以再建立连乘函数。

```julia
# 连乘函数，输入为数组，输出为数组中所有数的乘积
function sumMulti(x)
    if length(x) == 1
        return x
    else
        return x[1]*sumMulti(x[2:end])
    end
end
```

Lagrange插值公式中的多项式连乘形式单一，相比之下更像是多项式的两种表示形式的转换，所以再建立针对Lagrange插值公式的多项式转化函数

```julia
#将(x-x1)(x-x2)...(x-xn)化成an*x^n形式,A[1]为最低位(常数项)
#输入为[x1,x2,...,xn]输出为[a1,a2,...,an]
function polySimplify(x)
    if length(x) == 1
        return [-x[1],1]
    else
        return polyMulti([-x[1],1],polySimplify(x[2:end]))
    end
end
```

至此，可以创建Lagrange插值函数。

```julia
# Language插值函数
# x y: 维度相同的列变量
# A: 多项式系数，A[1]为最低位
using LinearAlgebra         #建立单位矩阵需要使用线性代数包
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
```

函数中的`xMat`为矩阵

$$\begin{bmatrix}
1 & x_1-x_0 & \ldots & x_n-x_0\\
x_0-x_1 & 1 & \ldots & x_n-x_1\\
& & \vdots & \\
x_0-x_n & x_1-x_n & \ldots & 0\\
\end{bmatrix}
$$

`polyMat`为多项式$\omega_{n+1}(x)$。

在命令行中调用得

```julia
julia> include("interpolation.jl")
Lagrange (generic function with 1 method)

julia> x = [i for i = 1:9];

julia> y = x.^8 + x.^6 + x.^2 .+ 1;

julia> Lagrange(x,y)
9-element Array{Float64,1}:
  1.0
  4.470348358154297e-8
  0.9999999850988388
 -1.4901161193847656e-8
 -7.450580596923828e-9
  2.7939677238464355e-9
  1.0
  0.0
  1.0

```

结果表明，其8次项、6次项、2次项和0次项分别为1或者接近于1，其他项分别为0或者接近于0，可见拟合结果尚可。通过`Plots`包对原始数据和拟合结果进行比较

```julia
julia> using Plots
julia> y1 = y;
julia> y2 = [sum([a[i]*x[j]^(i-1) for i=1:9]) for j = 1:9];
julia> plot(x,[y1,y2],st=[:scatter :line])
julia> savefig("Lagrange.png")
```
![Lagrange](img\Lagrange.png)

可见拟合结果与原始数据几乎一致。

## Newton插值

### 差商
对于一列$x_0,x_1,\ldots,x_n$，若与$y_0,y_1,\ldots,y_n$通过映射$f$一一对应，则定义比值

$$f[x_0,x_1]=\frac{f(x_1)-f(x_0)}{x_1-x_0}$$

为$f(x)$关于节点$x_0,x_1$的一阶差商。

类似地，

$$f[x_0,x_1,x_2]=\frac{f[x_0,x_2]-f[x_0,x_1]}{x_2-x_1}$$

为二阶差商，

$$f[x_0,x_1,\ldots,x_k]=\frac{f[x_0,x_1,\ldots,x_{k-2},x_k]-f[x_0,x_1,\ldots,x_{k-1}]}{x_k-x_{k-1}}$$

为$f(x)$关于节点$x_0,x_1,\ldots,x_k$的$k$阶差商。

$k$阶差商可表示为函数值$f(x_0),f(x_1),\ldots,f(x_n)$的线性组合，即

$$f[x_0,x_1,\ldots,x_k]=\displaystyle\sum_{j=0}^k\frac{f(x_j)}{(x_j-x_0)\ldots(x_j-x_{j-1})(x_j-x_{j+1})\ldots(x_j-x_k)}
$$

则

$$
f[x_0,x_1,\ldots,x_k]=\frac{f[x_1,x_2,\ldots,x_k]-f[x_0,x_1,\ldots,x_{k-1}]}{x_k-x_0}
$$

则可通过递归实现两个数组的差商

```julia
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
```

### Newton插值

根据差商定义，可得

$$\begin{aligned}
f(x) &= f(x_0) + f[x,x_0](x-x_0)\\
f[x,x_0] &= f[x_0,x_1] + f[x,x_0,x_1](x-x_1)\\
&\vdots\\
f[x,x_0,\ldots,x_{n-1}] &= f[x_0,x_1,\ldots,x_n] + f[x,x_0,x_1,\ldots,x_n](x-x_n)
\end{aligned}
$$

即

$$\begin{aligned}
f(x) =& f(x_0) + f[x_0,x_1](x-x_0)+ f[x_0,x_1,x_2](x-x_0)(x-x_1)\\
&+\ldots+f[x_0,x_1,\ldots,x_n](x-x_0)(x-x_1)\ldots(x-x_{n-1})\\
&+f[x_0,x_1,\ldots,x_n]\omega_{n+1}(x)\\
=&N_n(x)+R_n(x)    
\end{aligned}
$$

其中，

$$\begin{aligned}
N_n(x) =& f(x_0) + f[x_0,x_1](x-x_0)+ f[x_0,x_1,x_2](x-x_0)(x-x_1)+\\
&\ldots+f[x_0,x_1,\ldots,x_n](x-x_0)(x-x_1)\ldots(x-x_{n-1})\\
R_n(x)=&f[x_0,x_1,\ldots,x_n]\omega_{n+1}(x)
\end{aligned}
$$

$N_n(x)$即为牛顿插值公式，且

$$
N_{n+1}(x) = N_{n}(x) + f[x_0,x_1,\ldots,x_{n+1}]\omega_{n+1}(x)
$$

可快速写为Julia

```julia
# Newton插值函数
function Newton(x,y)
    n = length(x)
    A = zeros(n)
    A[1] = y[1]
    for i = 2:n
        p = DiffQuotient(x[1:i],y[1:i])*polySimplify(x[1:i-1])
        A[1:length(p)] += p
    end
    return A
end
```

当然，无论差商还是连乘运算，都会在运算的过程中产生大量的重复，上述代码中并未有效利用已经计算出的值，所以性能上并不占优。因此，我们对Newton插值进行更改

```julia
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
```

调用并验证

```julia
julia> include("interpolation.jl");
julia> x = [i for i = 1:9];
julia> y = x.^8 + x.^6 + x.^2 .+ 1;

julia> Newton(x,y)
9-element Array{Float64,1}:
 1.0
 0.0
 1.0
 0.0
 0.0
 0.0
 1.0
 0.0
 1.0
```

## Hermite插值

无论是Newton插值还是Lagrange插值，都只能在数值本身上满足插值函数与数据节点的重合，Hermite插值则要求其导数值相等。

设在节点$a\leqslant x_0\leqslant x_1 \leqslant\ldots\leqslant x_n\leqslant b$上，$y_j=f(x_j),m_j=f'(x_j),j=0,1,\ldots,n$，则插值多项式的条件为

$$
H(x_j)=y_j,H'(x_j)=m_j,j=0,1,\ldots,n
$$

总共有$2n+2$个等式，可唯一确定一个次数不大于$2n+1$的多项式$H_{2n+1}(x)$，设其表达式可通过基函数$\alpha_j,\beta_j$写成

$$
H_{2n+1}(x)=\displaystyle\sum_{j=0}^n[y_j\alpha_j+m_j\beta_j]
$$

则$\alpha_j,\beta_j$需要满足

$$\left\{\begin{aligned}
\alpha_j(x_k) &= \delta_{jk}\\
\alpha_j'(x_k)&= 0\\
\beta_j(x_k) &= 0\\
\beta_j'(x_k) & = \delta_{jk}
\end{aligned}\right.
$$

两个基函数的条件与Lagrange插值中的基函数相似，由于我们的目标函数为$2n+1$次，所以假设

$$\left\{\begin{aligned}
\alpha_j =  (ax+b)l_j^2(x)\\
\beta_j =  (cx+d)l_j^2(x)
\end{aligned}\right.$$

结合二者所满足的基函数条件，可得

$$\left\{\begin{aligned}
\alpha_j(x_j)&=(ax_j+b)l_j^2(x_j)=1\\
\alpha_j'(x_j)&=l_j(x_j)[al_j(x_j)+2(ax_j+b)l_j'(x_j)]=0\\
\beta_j(x_j)&=(cx_j+d)l_j^2(x_j)=0\\
\beta_j'(x_j)&=l_j(x_j)[cl_j(x_j)+2(cx_j+d)l_j'(x_j)]=1
\end{aligned}\right.
$$

解得

$$\left\{\begin{aligned}
\alpha_j &=  [1-2（x-x_j)\displaystyle\sum_{k=0,k\not=j}^n\frac{1}{x_j-x_k}]l_j^2(x)\\
\beta_j &=  (x-x_j)l_j^2(x)
\end{aligned}\right.$$

令$s_j=\displaystyle\sum_{k=0,k\not=j}\frac{1}{x_j-x_k}$,则Hermite插值公式可写为

$$
H_{2n+1}(x)=\displaystyle\sum_{j=0}^n\{
    [y_j+2s_jy_jx_j-x_jm_j]\cdot l_j^2(x)
    +[-2s_jy_j+m_j]\cdot xl_j^2(x)
    \}
$$

其代码为

```julia
# Hermite插值函数
# x,y分别为自变量、因变量，m为导数
function Hermite(x,y,m)
    n = length(x)
    unitMat = Matrix{Float64}(I,n,n)
    xMat = x .- x' + unitMat
    xInverse = 1./xMat - unitMat
    A = zeros(n*2+1)
    for i = 1:n
        quot = polyDiv(polyMat,[-x[i],1])[1]
        lag = quot ./ sumMulti(xMat[i,:])  #Lagrange基函数
        lag = polyMulti(lag,lag)
        sumInverse = sum(xInverse[i,:])
        a0 = (1+2*x[i]*sumInverse)*y[i]-x[i]*m[i]
        a1 = -2*y[i]+m[i]
        α = 1-2
        A += polyMulti(lag,[a0,a1])
    end
    return A
end
```

验证

```julia
julia> include("interpolation.jl");
julia> x = [i for i = 1:9];
julia> y = x.^8 + x.^6 + x.^2 .+ 1;
julia> m = [rand() for i = 1:9] 
julia> a = Hermite(x,y,m);

julia> y1 = y
julia> y2 = [sum([a[i]*x[j]^(i-1) for i=1:18]) for j = 1:9]
julia> plot(x,[y1,y2],st=[:scatter :line])
julia> savefig("hermite.png")
```

![Hermite](img\Hermite.png)

当然，实际上由于我们引入了一组随即变量作为没一点的导数，我们所拟合出的多项式函数自然与输入的多项式不同

```julia
julia> f(x) = sum([a[i]*x^(i-1) for i=1:18])
julia> plot(f,xlims=(1,9))
julia> savefig("realfig.png")
```

![realfig](img\realfig.png)

## 样条插值

### 高阶多项式拟合误差
由于在Hermite多项式拟合的过程中，在采用多项式函数生成$y$后，简单地通过随机值的方式生成了节点处的导数值，所以很难评价其拟合结果。

现我们通过多项式对函数$f(x)=\frac{1}{1+x^2}$进行拟合

```julia
julia> x = [i for i = -5:5];
julia> f(x) = 1/(1+x^2);
julia> df(x) = -2*x/(1+x^2)^2       #f(x)的导函数
julia> y = f.(x);                   #由于x为向量，所以函数括号前加.
julia> m = df.(x);
julia> a = Hermite(x,y,m)           #Hermite插值
julia> fHermite(x) = sum([a[i]*x^(i-1) for i=1:22])
julia> plot([f,fHermite])           #绘制函数f和fHermite
julia> savefig("errFit.png")        #保存图片
```

![errFit](img\errFit.png)

如图可知，高阶多项式函数在对某些数据进行插值的过程中可能会在局部拟合完美，而在另外的部分具有较大误差，这是Runge发现的，所以被称为Runge现象。如果能够在不同的位置通过分段的形式对数据进行拟合，或许能够克服这一问题。

### 三弯矩法

若函数$S(x)$在$[a,b]$区间内二阶可导，且在每个小区间$[x_k,x_{k+1}$上是三次多项式，其中$a\leqslant x_0\leqslant x_1 \leqslant\ldots\leqslant x_n\leqslant b$为给定节点，则称$S(x)$是节点上的三次样条函数，若在节点$x_j$上给定函数值$y_j=f(x_j),j=0,1,\ldots,n$，且$S(x_j)=y_j$，则称$S(x)$为三次样条插值函数。

对于任意一个小区间来说，三次插值需要有四个待定系数，除了端点处所对应的函数值$S_j(x_{j-1})=y_{j-1},S_j(x_j)=y_j$之外，还需要有两个等式共同构成边界条件。常见的边界条件有三种：

1. 已知两端的一阶导数
2. 已知两端的二阶导数
3. 周期边界条件

现假设我们已知任意点的二阶导数，记$S''(x_j)=M_j,h_j=x_j-x_{j-1},j=0,1,\ldots,n$，则$S''(x)$在区间$[x_{j-1},x_j]$之间是$x$的线性函数，且有$S''_j(x_{j-1})=M_{j-1},S''_j(x_j)=M_j$，通过线性插值，可将其表示为

$$
S''_j(x)=\frac{x_j-x}{h_j}M_{j-1}+\frac{x-x_{j-1}}{h_j}M_j
$$

对其积分可得

$$
S_j(x) = \frac{(x_j-x)^3}{6h_j}M_{j-1}+\frac{(x-x_{j-1})^3}{6h_j}M_j+C_1x+C_2
$$

将$S_j(x_{j-1})=y_{j-1},S_j(x_j)=y_j$代入，整理可得

$$
S_j(x) = \frac{(x_j-x)^3}{6h_j}M_{j-1}+\frac{(x-x_{j-1})^3}{6h_j}M_j+(y_{j-1}-\frac{M_{j-1}h_j^2}{6})\frac{x_j-x}{h_j}+(y_j-\frac{M_jh_j^2}{6})\frac{x-x_{j-1}}{h_j}
$$

这个表达式是天然的差分形式，可以直接写出相应的代码，在此不再列出。一般来说，任意一点的二阶导数对于已知数据而言仍旧是一个未知量，需要我们通过分段函数的连续特性进行求解，即要求$S'(x_j-0)=S'(x_j+0)$,求导可得

$$
S'_j(x)=-\frac{(x_j-x)^2}{2h_j}M_{j-1}+\frac{(x-x_{j-1})^2}{2h_j}M_j+\frac{y_j-y_{j-1}}{h_j}-\frac{(M_j-M_{j-1})h_j}{6}
$$

其左右端点为

$$\left\{\begin{aligned}
S'_j(x_j-0)&=\frac{h_j}{2}M_j+\frac{y_j-y_{j-1}}{h_j}-\frac{(M_j-M_{j-1})h_j}{6}\\
S'_j(x_{j-1}+0)&=-\frac{h_j}{2}M_{j-1}+\frac{y_j-y_{j-1}}{h_j}-\frac{(M_j-M_{j-1})h_j}{6}
\end{aligned}\right.
$$

$$\to\left\{\begin{aligned}
S'_j(x_j-0)&=\frac{h_jM_{j-1}}{6}+\frac{h_j}{3}M_j+\frac{y_j-y_{j-1}}{h_j}\\
S'_{j+1}(x_j+0)&=-\frac{h_{j+1}M_j}{3}-\frac{h_{j+1}}{6}M_{j+1}+\frac{y_{j+1}-y_j}{h_{j+1}}\\
\end{aligned}\right.
$$

根据连续性条件，上面两个等式应该处处相等，则该等式可整理为

$$
\mu_jM_{j-1}+2M_j+\lambda_jM_{j+1}=d_j, \quad j=1,2,\ldots,n-1
$$

各参数分别为

$$\left\{\begin{aligned}
\lambda_j &=\frac{h_{j+1}}{h_j+h_{j+1}}\\
\mu_j &=1-\lambda_j=\frac{h_j}{h_j+h_{j+1}}\\
d_j &=6f[x{j-1,x_j,x_{j+1}}]
\end{aligned}\right.
$$

由于在力学上$M$被解释为梁在截面上的弯矩，所以上式被称为三弯矩方程。三弯矩方程再加上此前提到的两个边界条件，即可求出$M$所对应的唯一解。

#### 第一类边界条件

例如，若给定第一类边界条件，即$S'(x_0)=y_0',S'(x_n)=y_n'$，则可得

$$\left\{\begin{aligned}
S'_n(x_n)&=\frac{h_nM_{n-1}}{6}+\frac{h_n}{3}M_n+\frac{y_n-y_{n-1}}{h_n}=y_n'\\
S'_1(x_0)&=-\frac{h_1M_0}{3}-\frac{h_1}{6}M_1+\frac{y_1-y_0}{h_1}=y_0'\\
\end{aligned}\right.
$$

$$\to\left\{\begin{aligned}
M_{n-1}+2M_n&=\frac{6}{h_n}(y_n'-\frac{y_n-y_{n-1}}{h_n})&\to d_n\\
2M_0+M_1&=\frac{6}{h_1}(-y_0'+\frac{y_1-y_0}{h_1})&\to d_0
\end{aligned}\right.
$$

将边界条件与三弯矩方程联立在一起，可得到确定$n+1$个参数的方程组，写成矩阵形式为

$$\begin{bmatrix}
2&1&&&&\\
\mu_1&2&\lambda_1&&&\\
&\mu_2&2&\lambda_2&&\\
&&\ddots&\ddots&\\
&&&\mu_{n-1}&2&\lambda_{n-1}\\
&&&&1&2
\end{bmatrix}
\begin{bmatrix}
M_0\\M_1\\\vdots\\M_{n-1}\\M_{n}  
\end{bmatrix}=
\begin{bmatrix}
d_0\\d_1\\\vdots\\d_{n-1}\\d_{n}  
\end{bmatrix}
$$

#### 第二类边界条件

第二类边界条件为$S''(x_0)=y_0'',S''(x_n)=y_n''$，其矩阵方程可表示为

$$\begin{bmatrix}
2&\lambda_1&&&\\
\mu_2&2&\lambda_2&&\\
&\ddots&\ddots&\\
&&\mu_{n-2}&2&\lambda_{n-2}\\
&&&\mu_{n-1}&2
\end{bmatrix}
\begin{bmatrix}
M_1\\M_2\\\vdots\\M_{n-2}\\M_{n-1}  
\end{bmatrix}=
\begin{bmatrix}
d_1-\mu_1y_0''\\d_2\\\vdots\\d_{n-2}\\d_{n-1}-\lambda_{n-1}y_n''  
\end{bmatrix}
$$

#### 第三类边界条件

第三类边界条件为周期条件，

$$S''(x_0+0)=S''(x_n-0),S'(x_0+0)=S'(x_n-0)$$

$$\to M_0=M_n,\quad \lambda_nM_1+\mu_nM_{n-1}+2M_n=d_n$$

其中，

$$\lambda_n=\frac{h_1}{h_1+h_n},\mu_n=1-\lambda_n=\frac{h_n}{h_1+h_n}$$

$$d_n=\frac{6}{h_1+h_2}(\frac{y_1-y_0}{h_1}-\frac{y_n-y_{n-1}}{h_n})$$

其矩阵方程为

$$\begin{bmatrix}
2&\lambda_1&&&\mu_1 \\
\mu_2&2&\lambda_2&&\\
&\ddots&\ddots&\\
&&\mu_{n-1}&2&\lambda_{n-1}\\
\lambda_n&&&\mu_n&2
\end{bmatrix}
\begin{bmatrix}
M_1\\M_2\\\vdots\\M_{n-1}\\M_n  
\end{bmatrix}=
\begin{bmatrix}
d_1\\d_2\\\vdots\\d_{n-1}\\d_n  
\end{bmatrix}
$$

求解过程代码如下

```julia
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

    # 生成M矩阵
    for i in 1:nMatrix-1
        M[i,i+1] = λ[i]
        M[i+1,i] = μ[i]
    end

    return M,d
end
```

## 参考资料：

1. 李庆杨.数值分析第四版
2. 颜庆津.数值分析第三版
3. Kincaid.数值分析第三版