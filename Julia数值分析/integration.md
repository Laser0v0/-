# 数值微积分

## 数值微分

导数的定义为

$$f'(a)=\lim\limits_{h\to0}\frac{f(a+h)-f(a)}{h}$$

则对于某精度条件下，选择适当的$h$值，即可将导数表示为

$$\begin{aligned}
f'(a)\approx&\frac{f(a+h)-f(a)}{h}\\
\approx&\frac{f(a)-f(a-h)}{h}\\
\approx&\frac{f(a+h)-f(a-h)}{2h}\\
\end{aligned}
$$

如果我们得到的是一组一一对应的$x,y$值，那么可以对这组数据进行多项式插值，然后再求解多项式的导数。Lagrange插值函数的表达式为

$$
L_n(x)=\displaystyle\sum_{k=0}^ny_k\frac{\omega_{n+1}(x)}{(x-x_k)\omega'_{n+1}(x_k)}
$$

其中，

$$
\omega_{n+1}(x)=(x-x_0)(x-x_1)\ldots(x-x_n)$$

其两点形式为

$$
L_2(x)=y_1\frac{x-x_2}{x_1-x_2}+y_2\frac{x-x_1}{x_2-x_1}
$$

由此可得两点插值的导数为

$$
L_2'(x)=\frac{y_1}{x_1-x_2}+\frac{y_2}{x_2-x_1}=\frac{y_2-y_1}{h}
$$

显然这是个常量，相当于根本没插值，直接用的前向或者后向差分。

Lagrange插值的三点表达式为

$$
L_3(x)=\frac{(x-x_1)(x-x_2)}{(x_0-x_1)(x_0-x_2)}f(x_0)+\frac{(x-x_0)(x-x_2)}{(x_1-x_0)(x_1-x_2)}f(x_1)+\frac{(x-x_0)(x-x_1)}{(x_2-x_1)(x_2-x_0)}f(x_2)
$$

其导数为

$$
L_3'(x)=\frac{2x-x_1-x_2}{(x_0-x_1)(x_0-x_2)}f(x_0)+\frac{2x-x_0-x_2}{(x_1-x_0)(x_1-x_2)}f(x_1)+\frac{2x-x_0-x_1}{(x_2-x_1)(x_2-x_0)}f(x_2)
$$

由此可得$x_0,x_1,x_2$点的导数分别为

$$\left\{\begin{aligned}
L_3'(x_0)=&\frac{1}{2h}[-3f(x_0)+4f(x_1)-f(x_2)]\\
L_3'(x_1)=&\frac{1}{2h}[-f(x_0)+f(x_2)]\\
L_3'(x_2)=&\frac{1}{2h}[-f(x_0)+4f(x_1)+3f(x_2)]\\
\end{aligned}\right.
$$

此即三点插值下的数值求导公式，通过同样的构造方式可以获取更多点数插值下的求导公式，在此便不再详述了。

## Newton-Cotes公式

定积分的定义式具有十分鲜明的数值特征

$$
\int_{a}^{b}f(x)\text{d}x=\lim_{n\to\infty}\displaystyle\sum_{k=1}^{n}f(a+\frac{k}{n}(b-a))\frac{b-a}{n}
$$

当$n\not\to\infty$时，令$h=\frac{b-a}{n}$，则上式变为

$$
\int_{a}^{b}f(x)\text{d}x\approx\displaystyle\sum_{k=1}^{n}f(a+kh)h
$$

可以十分方便地写成

```julia
# 无脑的数值积分算法
function nativeInt(f,a,b,n)
    h = (b-a)/n
    integ = 0
    for k = 1:n
        integ += f(a+k*h)*h
    end
    return integ
end
```

验证

```julia
julia> include("integration.jl");
julia> f(x) = 2*x^2 + 1     #定义被积函数
julia> nativeInt(f,0,1,20)  #积分区间为[0,1]，积分值为5/3
1.7175000000000002          #当步长为20的时候，误差很大

julia> nativeInt(f,0,1,100) #步长为100时，误差降到了1%
1.6766999999999999
```

可见其精度不怎么样，让牛顿那些人手算100次也的确有些强人所难了，所以牛顿他们就想办法在提高精度的前提下降低运算量。

考虑到定积分定义式的特点，令$x_k=a+k\frac{(b-a)}{n}$，为函数上的某些节点，则积分值可以表示为积分节点函数值$f(x_k)$的加权和，其权系数取决于当前点的斜率，定义为求积系数$A_k$。

$$
\int_{a}^{b}f(x)\text{d}x\approx\displaystyle\sum_{k=1}^{n}A_kf(x_k)
$$

如果$f(x_k)$为已经确定形式的多项式，由于多项式积分具有精确的表达式，所以可以确定最佳的$A_k$使得计算过程简化至最小计算量并且有最佳精度。于是我们马上想到了插值算法，如果令$L_n(x)$为插值函数，则可构造原积分的近似积分

$$
I_n=\int_a^bL_n(x)\text{d}x
$$

其中，

$$
L_n(x)=\displaystyle\sum_{k=0}^nf(x_k)l_k(x)
$$


通过插值积分来构造的插值型积分公式

$$
I_n=(b-a)\displaystyle\sum_{k=0}^{n}C^{(n)}_kf(x_k)
$$

被称为$\textbf{Newton-Cotes公式}$，其中$C^{(n)}_k$为$\textbf{Cotes系数}$。

将lagrange插值函数代入可得

$$
(b-a)\displaystyle\sum_{k=0}^{n}C^{(n)}_kf(x_k)=\displaystyle\sum_{k=0}^nf(x_k)\int_a^bl_k(x)\text{d}x
$$

即

$$\begin{aligned}
&&(b-a)C^{(n)}_k&=\int_a^bl_k(x)\text{d}x\\
\to&&C^{(n)}_k&=\frac{1}{b-a}\int_a^b\frac{\omega_{n+1}(x)}{(x-x_k)\omega'_{n+1}(x_k)}\text{d}x\\
&&\xrightarrow{x=a+hz}&=\int_0^n\frac{\omega_{n+1}(a+hz)}{(a+hz-x_k)\omega'_{n+1}(x_k)}\text{d}x\\

\end{aligned}
$$

其中$x_k$如前文所定义$x_k=a+k\frac{(b-a)}{n}=a+hk$，

$$\left\{\begin{aligned}
\omega_{n+1}(x)&=(x-x_0)(x-x_1)\ldots(x-x_n)\\
\omega'_{n+1}(x_k)&=(x_k-x_0)\ldots(x_k-x_{k-1})(x_k-x_{k+1})\ldots(x_k-x_n)
\end{aligned}\right.
$$

化简可得

$$
C^{(n)}_k=\frac{(-1)^{n-k}}{nk!(n-k)!}\int_0^n\frac{\Omega_{n+1}(z)}{z-k}\text{d}z
$$

其中

$$
\Omega_{n+1}(z)=(z-0)(z-1)\ldots(z-n)
$$

所以，数值积分被转化成求Cotes系数的问题，若将Cotes系数进行数值求解，那么有三个问题需要解决，一是阶乘，二是$\Omega$函数，三是多项式积分公式。

其中，阶乘几乎是我们在学习编程语言过程中，学习递归的第一个案例，几乎没有人没写过，没有什么太大的花样。

```julia
# 阶乘
function factorial(n)
    return  n<=1 ? 1 : n*factorial(n-1)
end
```

$\Omega$函数在此前的插值算法中曾经写过，不过在Newton-Cotes公式中，其输入为一个自然数而非一组系数，所以需要做一些改进。对于多项式的表示则无需改变，通过一个数组来表示多项式即可。注意多项式除法的输出为两项，分别是值和余数。

```julia
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

多项式积分亦只需变动一位，即

$$
\int a_kx^k\text{d}x=\frac{a_k}{k+1}x^{k+1}
$$

易写为

```julia
#多项式积分，输入为([a0,a1,...,an],aStart,aEnd)
#输出为[b0,b1,...,bn+1]
function polyInf(a,as,ae)
    #因为积分后位数升1，所以幂数从1开始
    return sum([(ae^i-as^i)*a[i]/i for i in 1:length(a)])    
end
```

至此，我们就能够得到Cotes系数

```julia
# 获取Cotes系数
function getCotes(n,k=[])
    if k == []
        return [getCotes(n,i) for i in 0:n]
    end
    kerr = polySimplify([i for i in 0:n if i != k])
    cotes = (-1)^(n-k)/n/factorial(k)/factorial(n-k)
    return cotes*polyInf(kerr,0,n)
end
```

验证

```julia
julia> for n = 1:5
       print(getCotes(n))
       print('\n')
       end
[0.5, 0.5]
[0.16666666666666663, 0.6666666666666667, 0.16666666666666663]
[0.125, 0.375, 0.375, 0.125]
[0.0777777777777775, 0.3555555555555566, 0.13333333333333286, 0.3555555555555543, 0.07777777777777779]
[0.06597222222222247, 0.26041666666666285, 0.17361111111110858, 0.17361111111110858, 0.26041666666666285, 0.06597222222222172]
```

与Cotes系数表比较，可见结果是对的。

|n||
|-|-|-|-
|1|$\frac{1}{2}\quad\frac{1}{2}$
|2|$\frac{1}{6}\quad\frac{4}{6}\quad\frac{1}{6}$
|3|$\frac{1}{8}\quad\frac{3}{8}\quad\frac{3}{8}\quad\frac{1}{8}$
|4|$\frac{7}{90}\quad\frac{16}{45}\quad\frac{2}{15}\quad\frac{16}{45}\quad\frac{7}{90}$
|5|$\frac{19}{288}\quad\frac{25}{96}\quad\frac{25}{144}\quad\frac{25}{144}\quad\frac{25}{96}\quad\frac{19}{288}$

将Cotes系数代入公式

$$
I_n=(b-a)\displaystyle\sum_{k=0}^{n}C^{(n)}_kf(x_k)
$$

则当$n=1$时，可得到梯形公式 

$$I_1=(b-a)\frac{b-a}{2}[f(a)+f(b)]$$

当$n=2$时，得到$\textbf{Simpson公式}$

$$I_2=(b-a)\frac{b-a}{6}[f(a)+4f(\frac{a+b}{2})+f(b)]$$

当$n=4$时，被特称为$\textbf{Cotes公式}$

$$I_4=(b-a)\frac{b-a}{90}
[7f(x_0)+32f(x_1)+12f(x_2)+32f(x_3)+7f(x_4)]$$

则对于任意阶的Newton-Cotes公式，可以写为

```julia
# Newton-Cotes公式
function NewtonCotes(f,a,b,n)
    cotes = getCotes(n)
    return sum([cotes[i]*f(a+(b-a)/n*(i-1)) for i in 1:n+1])
end
```

验证

```julia
julia> include("integration.jl");
julia> f(x) = x^2+2*x+3;
julia> NewtonCotes(f,0,1,3)
4.333333333333333
```

## 复化求积法

Lagrange插值法所存在的问题在于，高阶插值可能会导致局部最优，但其他区域误差很大，基于Lagrange插值的Newton-Cotes公式也有类似的特征，当阶数较大时会导致求解结果不稳定。此外，不同阶数的Newton-Cotes公式针对于不同阶数的多项式拟合精度不同，对于指数函数的无穷阶数，当拟合区间较大时，必然存在精度上的灾难。

```julia
julia> f(x) = x^2+2*x+3;
julia> NewtonCotes(f,0,1,30)
9.752372644513475e25
julia> NewtonCotes(exp,0,10,5)-(exp(10)-1) 
-19714.53098399712
```

解决Lagrange插值中Runge现象的方法是样条插值，其基本思想是对定义域进行分段，在每个小段进行低阶拟合，其总拟合效果要优于对所有区域的高阶拟合。同理，对于Newton-Cotes公式也可以对积分区域进行细分，从而用更低阶的Cotes系数在更狭小的区域中进行求积，这种方法叫做复化求积法。

二阶复化求积法，即复化梯形公式为

$$
T_n=\displaystyle\sum^{n-1}_{k=0}\frac{h}{2}[f(x_k)+f(x_{k+1})]=\frac{h}{2}[f(a)+2\displaystyle\sum^{n-1}_{k=1}f(x_k)+f(b)]$$

记子区间$[x_k,x_{k+1}]$的中点为$x_{k+\frac{1}{2}}$，则复化Simpson公式为

$$\begin{aligned}
S_n&=\displaystyle\sum^{n-1}_{k=0}\frac{h}{6}[f(x_k)+4f(x_{k+\frac{1}{2}})+f(x_{k+1})]\\
&=\frac{h}{6}[f(a)+4\displaystyle\sum^{n-1}_{k=0}f(x_{k+\frac{1}{2}})+2\displaystyle\sum^{n-1}_{k=0}f(x_k)+f(b)]
\end{aligned}
$$

更高阶的Newton-Cotes公式亦然。

复化公式的好处是通过缩减区间长度，可以提高数据精度，因为对于连续函数来说，越短的区间意味着这个区间内的映射关系更趋近于线性。问题在于更细的区段划分意味着更大的运算量，所以实际应用的过程中往往求一步看一步，当不满足精度要求的时候尽量缩短步长，在满足精度之后则进行输出。

对于复化梯形公式来说，对于任意一段积分区间$[x_k,x_{k+1}]$，当我们对其进行二分时，其实只是增加了一个中点，则这个子区间上的积分值为

$$
\frac{h}{4}[f(x_k)+2f(x_{k+\frac{1}{2}})+f(x_{k+1})]
$$

则整个区间的求和公式变为

$$\begin{aligned}
T_{2n}&=\frac{h}{4}\displaystyle\sum^{n-1}_{k=0}[f(x_k)+f(x_{k+1})]+\frac{h}{2}\displaystyle\sum^{n-1}_{k=0}f(x_{k+\frac{1}{2}})\\
&=\frac{1}{2}Tn+\frac{h}{2}\displaystyle\sum^{n-1}_{k=0}f(x_{k+\frac{1}{2}})
\end{aligned}
$$

其Julia实现为

```julia
# 复化梯形公式,f为函数，a,b为积分区间，err为要求精度
function Trapz(f,a,b,err=0.01,nMax=10)
    n = 2
    nMax = 2^nMax
    h = (b-a)/2
    T = h/2*(f(a)+f(b)+2*f(a+h))
    myErr = err + 1                 #初始误差大于目标误差

    while myErr > err && n < nMax
        newX = [2*i-1 for i in 1:n]         #新的x序列
        n = n*2; h = h/2                    #更新n，h值
        new  = sum(f.(a.+h*newX))*h
        myErr = abs(T/2-new)
        T = T/2 + new           #新的梯形值
    end
    return T,myErr
end
```

测试一下指数函数，结果表明当划分区间为$2^10$时，其拟合精度小于1，优于Newton-Cotes公式。

```julia
julia> include("integration.jl");
julia> Trapz(exp,0,10)
(22025.640837203784, 0.5251238525834196)
julia> exp(10)-1
22025.465794806718
```

## Romberg算法

梯形复化求积虽然缩减了求积区间，从而极大提高了求积精度，但同时牺牲Newton-Cotes公式中的高阶项。最直观的方法当然是对Simpson公式或者Cotes公式进行复化求积，但从算法复用的角度考虑，如果能够基于梯形复化求积法构造出高阶Newton-Cotes公式，那将是即为便利的。

对比Simpson公式和梯形求积公式

$$\begin{aligned}
T=&(b-a)\frac{b-a}{2}[f(a)+f(b)]\\
\to T_2=&(b-a)\frac{b-a}{4}[f(a)+f(b)+2f(\frac{a+b}{2})]\\
S=&(b-a)\frac{b-a}{6}[f(a)+4f(\frac{a+b}{2})+f(b)]\\
\end{aligned}
$$

解得

$$
S_n = \frac{4T_2-T}{3}\to S_n=\frac{4T_{2n}-T_n}{3}
$$

同理可得复化Cotes公式和$\textbf{Romberg公式}$

$$\begin{aligned}
C_n =&\frac{16T_{2n}-T_n}{15}\\
R_n =&\frac{64C_{2n}-1_n}{63}\\
\end{aligned}
$$

所以，我们可以直接对梯形复化公式进行修改

```julia
# 复化梯形公式,f为函数，a,b为积分区间，err为要求精度
# flag为1，2，3，4分别表示梯形、Simpson、Cotes、Romberg算法
function Trapz(f,a,b,flag=1,err=0.01,nMax=10)
    flag = min(flag,nMax)       #
    nMax = 2^nMax

    n = 1;    h = b-a                       #初始化n,h
    T = h/2*(f(a)+f(b))
    Tn = zeros(flag)                        #存储梯形公式的值
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
        Tn[end] = T                         #更新梯形公式
    end

    if flag > 1
        Sn = (4*Tn[2:end]-Tn[1:end-1])/3
        Tn = [Tn[end],Sn[end]]
    end

    if flag > 2
        Cn = (16*Sn[2:end]-Sn[1:end-1])/15  #Cotes公式
        Tn = vcat(Tn,Cn[end])
    end

    if flag > 3
        Rn = (64*Cn[2:end]-Cn[1:end-1])/63  #Romberg公式
        Tn = vcat(Tn,Rn[end])
    end

    return Tn,myErr
end
```

验证

```julia
julia> include("integration.jl");
julia> Trapz(exp,0,10,4)[1]
4-element Array{Float64,1}:
 22025.640837203784
 22025.46579591959
 22025.465794806754
 22025.46579480671
julia> exp(10)-1
22025.465794806718
```

可见在相同积分步长的前提下，Romberg公式的精度优于Cotes优于Simpson优于梯形公式。

