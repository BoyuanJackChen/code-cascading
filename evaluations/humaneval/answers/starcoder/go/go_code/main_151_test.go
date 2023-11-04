package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
         "math" 
 )

func TestDoubleTheDifference(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(0, DoubleTheDifference([]float64{}))
    assert.Equal(25, DoubleTheDifference([]float64{5, 4}))
    assert.Equal(0, DoubleTheDifference([]float64{0.1, 0.2, 0.3}))
    assert.Equal(0, DoubleTheDifference([]float64{-10, -20, -30}))
    assert.Equal(0, DoubleTheDifference([]float64{-1, -2, 8}))
    assert.Equal(34, DoubleTheDifference([]float64{0.2, 3, 5}))
    lst := make([]float64, 0)
    odd_sum := 0
    var i float64
    for i = -99; i < 100; i+= 2 {
        lst = append(lst, float64(i))
        if math.Mod(i, 2) != 0 && i > 0 {
            odd_sum +=int(math.Pow(i, 2))
        }
    }
    assert.Equal(odd_sum, DoubleTheDifference(lst))
}
