package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestProdSigns(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(-9, ProdSigns([]int{1, 2, 2, -4}))
    assert.Equal(0, ProdSigns([]int{0, 1}))
    assert.Equal(-10, ProdSigns([]int{1, 1, 1, 2, 3, -1, 1}))
    assert.Equal(nil, ProdSigns([]int{}))
    assert.Equal(20, ProdSigns([]int{2, 4,1, 2, -1, -1, 9}))
    assert.Equal(4, ProdSigns([]int{-1, 1, -1, 1}))
    assert.Equal(-4, ProdSigns([]int{-1, 1, 1, 1}))
    assert.Equal(0, ProdSigns([]int{-1, 1, 1, 0}))
}
