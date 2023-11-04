package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestBelowThreshold(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, BelowThreshold([]int{1, 2, 4, 10}, 100))
    assert.Equal(false, BelowThreshold([]int{1, 20, 4, 10}, 5))
    assert.Equal(true, BelowThreshold([]int{1, 20, 4, 10}, 21))
    assert.Equal(true, BelowThreshold([]int{1, 20, 4, 10}, 22))
    assert.Equal(true, BelowThreshold([]int{1, 8, 4, 10}, 11))
    assert.Equal(false, BelowThreshold([]int{1, 8, 4, 10}, 10))
}
