package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestMinPath(t *testing.T) {
    assert := assert.New(t)
    assert.Equal([]int{1, 2, 1}, Minpath([][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 3))
    assert.Equal([]int{1}, Minpath([][]int{{5, 9, 3}, {4, 1, 6}, {7, 8, 2}}, 1))
    assert.Equal([]int{1, 2, 1, 2}, Minpath([][]int{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}, 4))
    assert.Equal([]int{1, 10, 1, 10, 1, 10, 1}, Minpath([][]int{{6, 4, 13, 10}, {5, 7, 12, 1}, {3, 16, 11, 15}, {8, 14, 9, 2}}, 7))
    assert.Equal([]int{1, 7, 1, 7, 1}, Minpath([][]int{{8, 14, 9, 2}, {6, 4, 13, 15}, {5, 7, 1, 12}, {3, 10, 11, 16}}, 5))
    assert.Equal([]int{1, 6, 1, 6, 1, 6, 1, 6, 1}, Minpath([][]int{{11, 8, 7, 2}, {5, 16, 14, 4}, {9, 3, 15, 6}, {12, 13, 10, 1}}, 9))
    assert.Equal([]int{1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6}, Minpath([][]int{{12, 13, 10, 1}, {9, 3, 15, 6}, {5, 16, 14, 4}, {11, 8, 7, 2}}, 12))
    assert.Equal([]int{1, 3, 1, 3, 1, 3, 1, 3}, Minpath([][]int{{2, 7, 4}, {3, 1, 5}, {6, 8, 9}}, 8))
    assert.Equal([]int{1, 5, 1, 5, 1, 5, 1, 5}, Minpath([][]int{{6, 1, 5}, {3, 8, 9}, {2, 7, 4}}, 8))
    assert.Equal([]int{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}, Minpath([][]int{{1, 2}, {3, 4}}, 10))
    assert.Equal([]int{1, 3, 1, 3, 1, 3, 1, 3, 1, 3}, Minpath([][]int{{1, 3}, {3, 2}}, 10))
}
