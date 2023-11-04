package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSumSquares(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(6, SumSquares([]int{1,2,3}))
    assert.Equal(14, SumSquares([]int{1,4,9}))
    assert.Equal(0, SumSquares([]int{}))
    assert.Equal(9, SumSquares([]int{1,1,1,1,1,1,1,1,1}))
    assert.Equal(-3, SumSquares([]int{-1,-1,-1,-1,-1,-1,-1,-1,-1}))
    assert.Equal(0, SumSquares([]int{0}))
    assert.Equal(-126, SumSquares([]int{-1,-5,2,-1,-5}))
    assert.Equal(3030, SumSquares([]int{-56,-99,1,0,-2}))
    assert.Equal(0, SumSquares([]int{-1,0,0,0,0,0,0,0,-1}))
    assert.Equal(-14196, SumSquares([]int{-16, -9, -2, 36, 36, 26, -20, 25, -40, 20, -4, 12, -26, 35, 37}))
    assert.Equal(-1448, SumSquares([]int{-1, -3, 17, -1, -15, 13, -1, 14, -14, -12, -5, 14, -14, 6, 13, 11, 16, 16, 4, 10}))
}
