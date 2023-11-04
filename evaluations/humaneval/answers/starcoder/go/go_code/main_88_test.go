package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestSortArray(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]int{}, SortArray([]int{}), "Error")
    assert.Equal([]int{5}, SortArray([]int{5}), "Error")
    assert.Equal([]int{0, 1, 2, 3, 4, 5}, SortArray([]int{2, 4, 3, 0, 1, 5}), "Error")
    assert.Equal([]int{6, 5, 4, 3, 2, 1, 0}, SortArray([]int{2, 4, 3, 0, 1, 5, 6}), "Error")
    assert.Equal([]int{1, 2}, SortArray([]int{2, 1}), "Error")
    assert.Equal([]int{0, 11, 15, 32, 42, 87}, SortArray([]int{15, 42, 87, 32, 11, 0}), "Error")
    assert.Equal([]int{23, 21, 14, 11}, SortArray([]int{21, 14, 23, 11}), "Error")
}
